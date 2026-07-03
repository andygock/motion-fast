from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .runner import process_input, resolve_input_paths
from .utils import die


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
        help="Use NVIDIA NVENC when encoding the review video.",
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
        help="Do not delete the existing events.csv output directory before running.",
    )

    parser.add_argument(
        "--no-clobber",
        action="store_true",
        help="Skip processing if the final review MP4 already exists beside the input file.",
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

    args = parser.parse_args()

    if args.width < 16:
        die("--width must be at least 16")

    if args.fps <= 0:
        die("--fps must be greater than zero")

    if args.speed <= 0:
        die("--speed must be greater than zero")

    if args.extract_workers < 0:
        die("--extract-workers must be zero or greater")

    if not shutil.which("ffmpeg"):
        die("ffmpeg was not found on PATH")

    if not shutil.which("ffprobe"):
        die("ffprobe was not found on PATH")

    input_paths = resolve_input_paths(args.inputs)

    for index, input_path in enumerate(input_paths, start=1):
        process_input(
            input_path,
            args,
            input_count=len(input_paths),
            input_index=index,
        )

