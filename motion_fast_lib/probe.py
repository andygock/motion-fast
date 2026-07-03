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
    data = json.loads(result.stdout)

    streams = data.get("streams", [])
    if not streams:
        die("No video stream found")

    stream = streams[0]

    width = int(stream["width"])
    height = int(stream["height"])

    duration_value = stream.get("duration") or data.get(
        "format", {}).get("duration")
    if duration_value is None:
        die("Could not determine video duration")

    duration = float(duration_value)

    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(
        stream.get("r_frame_rate")) or 0.0

    return VideoInfo(
        width=width,
        height=height,
        duration=duration,
        fps=fps,
    )


