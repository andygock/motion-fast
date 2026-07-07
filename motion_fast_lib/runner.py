from __future__ import annotations

import argparse
import glob
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from .analysis import detect_motion
from .events import merge_motion_frames, write_events_csv
from .probe import ffprobe_video
from .review import build_review_video, ffmpeg_encoder_usable, resolve_ffmpeg_threads
from .models import Event
from .utils import die, fmt_time


@dataclass
class ExistingOutputSummary:
    total: int
    will_not_overwrite: int
    will_overwrite: int


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            die(f"Output path exists and is not a directory: {output_dir}")

    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    events_csv = output_dir / "events.csv"
    if events_csv.exists() and not events_csv.is_file():
        die(f"Event log path exists and is not a file: {events_csv}")


def print_events(events: list[Event]) -> None:
    if not events:
        print("No merged events.")
        return

    print("Merged events")
    print("  #    Start          End            Duration       Frames   Peak changed")
    print("  ---  -------------  -------------  -------------  -------  ------------")

    for e in events:
        print(
            f"  {e.index:>3}  "
            f"{fmt_time(e.start_s):>13}  "
            f"{fmt_time(e.end_s):>13}  "
            f"{fmt_time(e.duration_s):>13}  "
            f"{e.motion_frames:>7}  "
            f"{e.peak_changed_pixels:>12,}"
        )

    total_duration = sum(e.duration_s for e in events)
    print()
    print(f"Events           : {len(events)}")
    print(f"Review duration  : {fmt_time(total_duration)} before speed-up")


def has_glob_chars(path: Path) -> bool:
    return any(char in str(path) for char in "*?[")


def resolve_input_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    skipped: list[tuple[Path, OSError]] = []

    for input_arg in inputs:
        if input_arg.exists():
            matches = [input_arg]
        elif has_glob_chars(input_arg):
            matches = [
                Path(match)
                for match in glob.glob(str(input_arg), recursive=True)
            ]
        else:
            die(f"Input file does not exist: {input_arg}")

        if not matches:
            die(f"Input pattern matched no files: {input_arg}")

        for match in sorted(matches, key=lambda p: str(p).lower()):
            try:
                resolved = match.resolve()
                exists = resolved.exists()
                is_file = resolved.is_file()
            except OSError as exc:
                skipped.append((match, exc))
                continue

            if not exists:
                die(f"Input file does not exist: {resolved}")
            if not is_file:
                die(f"Input path is not a file: {resolved}")
            if resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)

    for skipped_path, exc in skipped:
        print(f"Skipping inaccessible input: {skipped_path}")
        print(f"  {exc}")

    if not paths and skipped:
        die("No accessible input files were found.")

    return paths


def output_dir_for_input(input_path: Path, args: argparse.Namespace, input_count: int) -> Path:
    if args.out_dir is None:
        return input_path.with_name(f"{input_path.stem}_motion_review").resolve()

    output_dir = args.out_dir.resolve()
    if input_count == 1:
        return output_dir

    input_hash = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return output_dir / f"{input_path.stem}_{input_hash}_motion_review"


def existing_output_summary(
    input_paths: list[Path],
    args: argparse.Namespace,
) -> ExistingOutputSummary:
    total = 0
    will_not_overwrite = 0
    will_overwrite = 0
    input_count = len(input_paths)
    write_events = args.detect_only or args.write_events_csv

    for input_path in input_paths:
        output_dir = output_dir_for_input(input_path, args, input_count)
        review_mp4 = input_path.parent / f"review_{input_path.stem}.mp4"
        events_csv = output_dir / "events.csv"
        review_exists = review_mp4.exists()

        if not args.detect_only and review_exists:
            total += 1
            if args.no_clobber:
                will_not_overwrite += 1
            else:
                will_overwrite += 1

        if write_events and events_csv.exists():
            total += 1
            if args.no_clobber and not args.detect_only and review_exists:
                will_not_overwrite += 1
            else:
                will_overwrite += 1

    return ExistingOutputSummary(
        total=total,
        will_not_overwrite=will_not_overwrite,
        will_overwrite=will_overwrite,
    )


def print_existing_output_summary(
    input_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    summary = existing_output_summary(input_paths, args)

    print("Output preflight check (state as of now)")
    print(f"  Existing output files : {summary.total}")
    print(f"  Will not overwrite    : {summary.will_not_overwrite}")
    print(f"  Will overwrite        : {summary.will_overwrite}")
    print("Processing still checks file state again when each input starts.")
    print()


def process_input(
    input_path: Path,
    args: argparse.Namespace,
    *,
    input_count: int,
    input_index: int,
) -> None:
    if input_count > 1:
        print()
        print(f"Processing {input_index}/{input_count}: {input_path}")
        print()

    output_dir = output_dir_for_input(input_path, args, input_count)
    review_mp4 = input_path.parent / f"review_{input_path.stem}.mp4"
    write_events = args.detect_only or args.write_events_csv

    if args.no_clobber and not args.detect_only and review_mp4.exists():
        print(f"Skipping {input_path}: output already exists: {review_mp4}")
        return

    start_wall = time.time()

    if write_events:
        prepare_output_dir(output_dir)

    info = ffprobe_video(input_path)

    motion_frames = detect_motion(
        input_path=input_path,
        info=info,
        width=args.width,
        sample_fps=args.fps,
        pixel_threshold=args.pixel_threshold,
        motion_threshold=args.motion_threshold,
        min_consecutive=args.min_consecutive,
        use_cuda=not args.no_cuda_decode,
        keyframes_only=args.keyframes_only,
        color_detect=args.color_detect,
        timestamp_mode=args.timestamp_mode,
        debug_every=args.debug_every,
        quiet=args.quiet,
    )

    events = merge_motion_frames(
        motion_frames,
        duration=info.duration,
        pre_roll=args.pre_roll,
        post_roll=args.post_roll,
        merge_gap=args.merge_gap,
        min_event_duration=args.min_event_duration,
    )

    events_csv = output_dir / "events.csv"
    if write_events:
        write_events_csv(events_csv, events)

    print_events(events)
    if write_events:
        print()
        print(f"Event log written: {events_csv}")

    if args.detect_only:
        elapsed = time.time() - start_wall
        print()
        print("Detect-only mode enabled. No review video was built.")
        print(f"  Elapsed time     : {fmt_time(elapsed)}")
        return

    if not events:
        elapsed = time.time() - start_wall
        print()
        print("No motion events found, so no review video was created.")
        print(f"  Elapsed time     : {fmt_time(elapsed)}")
        return

    print()
    print("Building review video")

    ffmpeg_threads = resolve_ffmpeg_threads(args.extract_workers)
    if ffmpeg_threads > 1 and not (args.copy_video and args.speed == 1):
        print(f"  FFmpeg threads   : {ffmpeg_threads}")

    encode_required = not (args.copy_video and args.speed == 1)
    use_nvenc = args.encoder == "nvenc" or args.nvenc
    if encode_required and args.encoder == "auto" and not args.nvenc:
        use_nvenc = ffmpeg_encoder_usable("h264_nvenc")

    if encode_required:
        print(f"  Encoder          : {'h264_nvenc' if use_nvenc else 'libx264'}")

    print()
    print(f"Creating {review_mp4.name}")

    build_review_video(
        input_path=input_path,
        output_path=review_mp4,
        events=events,
        speed=args.speed,
        use_nvenc=use_nvenc,
        crf=args.crf,
        preset=args.preset,
        copy_video=args.copy_video,
        ffmpeg_threads=ffmpeg_threads,
    )

    total_review_duration = sum(e.duration_s for e in events) / args.speed
    elapsed = time.time() - start_wall

    print()
    print("Done")
    print(f"  Review video     : {review_mp4}")
    if write_events:
        print(f"  Events CSV       : {events_csv}")
    print(f"  Events           : {len(events)}")
    print(f"  Elapsed time     : {fmt_time(elapsed)}")
    print(
        f"  Review duration  : {fmt_time(total_review_duration)} after speed-up")


