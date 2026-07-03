from __future__ import annotations

import csv
from pathlib import Path

from .models import Event, MotionFrame
from .utils import fmt_time


def merge_motion_frames(
    motion_frames: list[MotionFrame],
    *,
    duration: float,
    pre_roll: float,
    post_roll: float,
    merge_gap: float,
    min_event_duration: float,
) -> list[Event]:
    if not motion_frames:
        return []

    motion_frames = sorted(motion_frames, key=lambda x: x.time_s)

    groups: list[list[MotionFrame]] = []
    current_group: list[MotionFrame] = [motion_frames[0]]

    for mf in motion_frames[1:]:
        prev = current_group[-1]

        if mf.time_s - prev.time_s <= merge_gap:
            current_group.append(mf)
        else:
            groups.append(current_group)
            current_group = [mf]

    groups.append(current_group)

    events: list[Event] = []

    for group in groups:
        raw_start = group[0].time_s
        raw_end = group[-1].time_s

        start_s = max(0.0, raw_start - pre_roll)
        end_s = min(duration, raw_end + post_roll)

        if end_s < start_s:
            continue

        event_duration = end_s - start_s

        if event_duration < min_event_duration:
            continue

        peak = max(mf.changed_pixels for mf in group)

        events.append(
            Event(
                index=len(events) + 1,
                start_s=start_s,
                end_s=end_s,
                duration_s=event_duration,
                motion_frames=len(group),
                peak_changed_pixels=peak,
            )
        )

    return events


def write_events_csv(path: Path, events: list[Event]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "index",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "start_time",
                "end_time",
                "duration_time",
                "motion_frames",
                "peak_changed_pixels",
            ]
        )

        for e in events:
            writer.writerow(
                [
                    e.index,
                    f"{e.start_s:.3f}",
                    f"{e.end_s:.3f}",
                    f"{e.duration_s:.3f}",
                    fmt_time(e.start_s),
                    fmt_time(e.end_s),
                    fmt_time(e.duration_s),
                    e.motion_frames,
                    e.peak_changed_pixels,
                ]
            )


