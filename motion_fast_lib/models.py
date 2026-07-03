from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float


@dataclass
class MotionFrame:
    time_s: float
    changed_pixels: int


@dataclass
class Event:
    index: int
    start_s: float
    end_s: float
    duration_s: float
    motion_frames: int
    peak_changed_pixels: int
