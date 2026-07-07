from __future__ import annotations

import json
from pathlib import Path

from .models import VideoInfo
from .utils import die, parse_rate, run_command


def ffprobe_video(input_path: Path) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(input_path),
    ]

    result = run_command(cmd, capture_output=True)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        die(f"Could not parse ffprobe output for {input_path}: {exc}")

    streams = data.get("streams", [])
    if not streams:
        die("No video stream found")

    stream = streams[0]

    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        die(f"Could not determine video dimensions for {input_path}: {exc}")

    duration_value = stream.get("duration") or data.get(
        "format", {}).get("duration")
    if duration_value is None:
        die(f"Could not determine video duration for {input_path}")

    try:
        duration = float(duration_value)
    except (TypeError, ValueError) as exc:
        die(f"Could not parse video duration for {input_path}: {exc}")

    try:
        fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(
            stream.get("r_frame_rate")) or 0.0
    except ValueError as exc:
        die(f"Could not parse video frame rate for {input_path}: {exc}")

    return VideoInfo(
        width=width,
        height=height,
        duration=duration,
        fps=fps,
    )


