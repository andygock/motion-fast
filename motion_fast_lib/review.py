from __future__ import annotations

import os
from pathlib import Path

from .models import Event
from .utils import run_command


def ffmpeg_escape_concat_path(path: Path) -> str:
    # FFmpeg concat demuxer accepts forward slashes on Windows.
    # Single quotes inside paths need escaping.
    s = path.as_posix()
    s = s.replace("'", "'\\''")
    return f"file '{s}'"


def resolve_ffmpeg_threads(requested_threads: int) -> int:
    if requested_threads > 0:
        return requested_threads

    return os.cpu_count() or 1


def build_review_concat_manifest(
    input_path: Path,
    events: list[Event],
) -> str:
    lines: list[str] = []
    for event in events:
        lines.append(ffmpeg_escape_concat_path(input_path))
        lines.append(f"inpoint {event.start_s:.3f}")
        lines.append(f"outpoint {event.end_s:.3f}")

    return "\n".join(lines) + "\n"


def build_review_video(
    input_path: Path,
    output_path: Path,
    events: list[Event],
    *,
    speed: float,
    use_nvenc: bool,
    crf: int,
    preset: str,
    copy_video: bool,
    ffmpeg_threads: int,
) -> None:
    concat_manifest = build_review_concat_manifest(input_path, events)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-protocol_whitelist",
        "file,pipe,fd",
        "-i",
        "-",
        "-an",
        "-sn",
        "-dn",
    ]

    if copy_video and speed == 1:
        cmd += [
            "-c:v",
            "copy",
        ]
    else:
        if speed != 1:
            cmd += [
                "-vf",
                f"setpts=PTS/{speed}",
            ]

        if ffmpeg_threads > 0:
            cmd += [
                "-filter_threads",
                str(ffmpeg_threads),
                "-threads",
                str(ffmpeg_threads),
            ]

        if use_nvenc:
            cmd += [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p7",
                "-cq",
                str(crf),
                "-pix_fmt",
                "yuv420p",
            ]
        else:
            cmd += [
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
            ]

    cmd += [
        str(output_path),
    ]

    run_command(cmd, capture_output=True, input_data=concat_manifest)


