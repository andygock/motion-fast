from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .runner import print_existing_output_summary, process_input, resolve_input_paths
from .utils import die


def process_input_captured(
    input_path: Path,
    args: argparse.Namespace,
    input_count: int,
    input_index: int,
) -> tuple[int, str, int]:
    buffer = io.StringIO()
    exit_code = 0
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            process_input(
                input_path,
                args,
                input_count=input_count,
                input_index=input_index,
            )
        except SystemExit as exc:
            exit_code = int(exc.code) if isinstance(exc.code, int) else 1
        except BaseException:
            exit_code = 1
            traceback.print_exc()

    return input_index, buffer.getvalue(), exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create condensed motion-review MP4 files from CCTV video files."
    )

    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Input video file(s) or glob pattern(s), for example 20260615-20260622.avi or /path/*.avi",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for events.csv when it is written. Default: <input_stem>_motion_review. With multiple inputs, this is used as a parent directory.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Analysis width in pixels. Smaller is faster. Default: 320",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Analysis sample rate for --all-frames mode. Default: 2",
    )

    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=30,
        help="Per-pixel brightness difference required to count as changed. Default: 30",
    )

    parser.add_argument(
        "--motion-threshold",
        type=int,
        default=200,
        help="Changed pixel count required to mark a frame as motion. Default: 200",
    )

    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=1,
        help="Number of consecutive motion frames required before accepting motion. Default: 1",
    )

    parser.add_argument(
        "--merge-gap",
        type=float,
        default=6.0,
        help="Merge motion detections separated by this many seconds or less. Default: 6",
    )

    parser.add_argument(
        "--pre-roll",
        type=float,
        default=0.5,
        help="Seconds to include before detected motion. Default: 0.5",
    )

    parser.add_argument(
        "--post-roll",
        type=float,
        default=0.5,
        help="Seconds to include after detected motion. Default: 0.5",
    )

    parser.add_argument(
        "--min-event-duration",
        type=float,
        default=1.0,
        help="Discard merged events shorter than this many seconds. Default: 1",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed up the review video. Example: --speed 4. Default: 1",
    )

    parser.add_argument(
        "--no-cuda-decode",
        action="store_true",
        help="Disable FFmpeg CUDA hardware decode during scanning.",
    )

    keyframe_group = parser.add_mutually_exclusive_group()
    keyframe_group.add_argument(
        "--keyframes-only",
        action="store_true",
        default=True,
        help="Scan only H.264 keyframes. This is the default. Much faster when keyframes are frequent, but may miss short motion between keyframes.",
    )

    keyframe_group.add_argument(
        "--all-frames",
        dest="keyframes_only",
        action="store_false",
        help="Scan sampled frames instead of only keyframes. Slower, but can catch motion between keyframes.",
    )

    parser.add_argument(
        "--color-detect",
        action="store_true",
        help="Compare RGB channels instead of grayscale so color-only changes can count as motion.",
    )

    parser.add_argument(
        "--nvenc",
        action="store_true",
        help="Use NVIDIA NVENC when encoding the review video. Same as --encoder nvenc.",
    )

    parser.add_argument(
        "--encoder",
        choices=("auto", "cpu", "nvenc"),
        default="auto",
        help="Encoder for review video re-encoding. auto uses NVENC when FFmpeg can run h264_nvenc, otherwise libx264. Default: auto",
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        help="Quality value for encoded review output. libx264 uses CRF. NVENC uses CQ. Default: 28",
    )

    parser.add_argument(
        "--preset",
        default="veryfast",
        help="libx264 preset if not using NVENC. Default: veryfast",
    )

    copy_group = parser.add_mutually_exclusive_group()
    copy_group.add_argument(
        "--copy-video",
        dest="copy_video",
        action="store_true",
        default=True,
        help="Build the review with stream copy. Default: enabled. Fastest, but cuts may be less accurate.",
    )

    copy_group.add_argument(
        "--reencode-clips",
        dest="copy_video",
        action="store_false",
        help="Re-encode the review instead of stream-copying it.",
    )

    parser.add_argument(
        "--extract-workers",
        type=int,
        default=0,
        help="FFmpeg thread count used when encoding the review. Has no effect for stream-copy mode. Default: 0 = auto (all logical CPUs).",
    )

    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Compatibility option. Event output directories are never deleted; existing unrelated files are kept.",
    )

    clobber_group = parser.add_mutually_exclusive_group()
    clobber_group.add_argument(
        "--no-clobber",
        dest="no_clobber",
        action="store_true",
        default=True,
        help="Skip processing if the final review MP4 already exists beside the input file. This is the default.",
    )

    clobber_group.add_argument(
        "--overwrite",
        dest="no_clobber",
        action="store_false",
        help="Overwrite an existing final review MP4.",
    )

    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only scan and write events.csv. Do not build a review video.",
    )

    parser.add_argument(
        "--write-events-csv",
        action="store_true",
        help="Write events.csv during normal review runs. Detect-only mode always writes events.csv.",
    )

    parser.add_argument(
        "--debug-every",
        type=float,
        default=1.0,
        help="Progress update interval in seconds. Default: 1",
    )

    parser.add_argument(
        "--timestamp-mode",
        choices=("exact", "approx"),
        default="exact",
        help="Timestamp mapping for keyframe-only scans. exact parses FFmpeg showinfo timestamps; approx skips that logging and spreads keyframes across duration for more speed. Default: exact",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live scan progress. Final summaries are still printed.",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of input files to process in parallel. Default: 1",
    )

    args = parser.parse_args()

    if args.width < 16:
        die("--width must be at least 16")

    if args.fps <= 0:
        die("--fps must be greater than zero")

    if not 0 <= args.pixel_threshold <= 255:
        die("--pixel-threshold must be between 0 and 255")

    if args.motion_threshold < 1:
        die("--motion-threshold must be at least 1")

    if args.min_consecutive < 1:
        die("--min-consecutive must be at least 1")

    if args.merge_gap < 0:
        die("--merge-gap must be zero or greater")

    if args.pre_roll < 0:
        die("--pre-roll must be zero or greater")

    if args.post_roll < 0:
        die("--post-roll must be zero or greater")

    if args.min_event_duration < 0:
        die("--min-event-duration must be zero or greater")

    if args.speed <= 0:
        die("--speed must be greater than zero")

    if not 0 <= args.crf <= 51:
        die("--crf must be between 0 and 51")

    if not args.preset.strip():
        die("--preset must not be empty")

    if args.extract_workers < 0:
        die("--extract-workers must be zero or greater")

    if args.debug_every <= 0:
        die("--debug-every must be greater than zero")

    if args.jobs < 1:
        die("--jobs must be at least 1")

    if args.nvenc:
        args.encoder = "nvenc"

    if not shutil.which("ffmpeg"):
        die("ffmpeg was not found on PATH")

    if not shutil.which("ffprobe"):
        die("ffprobe was not found on PATH")

    input_paths = resolve_input_paths(args.inputs)
    print_existing_output_summary(input_paths, args)

    if args.jobs == 1 or len(input_paths) == 1:
        for index, input_path in enumerate(input_paths, start=1):
            process_input(
                input_path,
                args,
                input_count=len(input_paths),
                input_index=index,
            )
        return

    worker_count = min(args.jobs, len(input_paths))
    print(f"Processing {len(input_paths)} inputs with {worker_count} parallel jobs")
    print("Live per-file progress is captured and printed when each job finishes.")

    failed = False
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                process_input_captured,
                input_path,
                args,
                len(input_paths),
                index,
            )
            for index, input_path in enumerate(input_paths, start=1)
        ]

        for future in as_completed(futures):
            _, output, exit_code = future.result()
            if output:
                print()
                print(output, end="" if output.endswith("\n") else "\n")
            if exit_code != 0:
                failed = True

    if failed:
        die("One or more inputs failed")

