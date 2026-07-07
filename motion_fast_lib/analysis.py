from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .models import MotionFrame, VideoInfo
from .utils import die, fmt_time, safe_even_height


SHOWINFO_TIME_RE = re.compile(r"pts_time:([^\s]+)")
PROGRESS_TIME_RE = re.compile(r"out_time_ms=(\d+)")


def read_showinfo_stderr(
    stderr_pipe,
    timestamp_queue: queue.Queue[float],
    progress_queue: queue.Queue[float],
    stderr_lines: list[str],
) -> None:
    for raw_line in iter(stderr_pipe.readline, b""):
        line = raw_line.decode(errors="replace").strip()
        match = SHOWINFO_TIME_RE.search(line)
        if match:
            try:
                timestamp_queue.put(float(match.group(1)))
            except ValueError:
                pass
            continue

        match = PROGRESS_TIME_RE.fullmatch(line)
        if match:
            progress_queue.put(int(match.group(1)) / 1_000_000)
            continue

        if line.startswith(
            (
                "frame=",
                "fps=",
                "stream_",
                "bitrate=",
                "total_size=",
                "out_time_us=",
                "out_time=",
                "dup_frames=",
                "drop_frames=",
                "speed=",
                "progress=",
            )
        ):
            continue

        if line:
            stderr_lines.append(line)


def estimate_keyframe_time(
    frame_index: int,
    keyframe_timestamps: list[float],
) -> float:
    if frame_index < len(keyframe_timestamps):
        return keyframe_timestamps[frame_index]

    if len(keyframe_timestamps) >= 2:
        keyframe_interval = (
            keyframe_timestamps[-1] - keyframe_timestamps[0]
        ) / max(1, len(keyframe_timestamps) - 1)
        return keyframe_timestamps[0] + frame_index * keyframe_interval

    # Without showinfo timestamps, return the keyframe ordinal. The caller can
    # rescale detections once the decoded keyframe count is known.
    return float(frame_index)


def drain_timestamp_queue(
    timestamp_queue: queue.Queue[float],
    keyframe_timestamps: list[float],
) -> None:
    while True:
        try:
            keyframe_timestamps.append(timestamp_queue.get_nowait())
        except queue.Empty:
            return


def drain_progress_queue(
    progress_queue: queue.Queue[float],
    fallback: float,
) -> float:
    latest = fallback
    while True:
        try:
            latest = progress_queue.get_nowait()
        except queue.Empty:
            return latest


def map_keyframe_motion_times(
    motion_frames: list[MotionFrame],
    *,
    keyframe_timestamps: list[float],
    decoded_keyframes: int,
    duration: float,
) -> list[MotionFrame]:
    if not motion_frames:
        return []

    if len(keyframe_timestamps) >= decoded_keyframes:
        return [
            MotionFrame(
                time_s=keyframe_timestamps[int(mf.time_s)],
                changed_pixels=mf.changed_pixels,
            )
            for mf in motion_frames
        ]

    if len(keyframe_timestamps) >= 2:
        return [
            MotionFrame(
                time_s=estimate_keyframe_time(int(mf.time_s), keyframe_timestamps),
                changed_pixels=mf.changed_pixels,
            )
            for mf in motion_frames
        ]

    scale = duration / max(1, decoded_keyframes - 1)
    return [
        MotionFrame(
            time_s=min(duration, mf.time_s * scale),
            changed_pixels=mf.changed_pixels,
        )
        for mf in motion_frames
    ]


def build_decode_command(
    input_path: Path,
    width: int,
    height: int,
    sample_fps: float,
    use_cuda: bool,
    keyframes_only: bool,
    color_detect: bool,
    timestamp_mode: str,
) -> list[str]:
    use_showinfo = keyframes_only and timestamp_mode == "exact"
    use_progress = keyframes_only and timestamp_mode == "approx"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info" if use_showinfo else "error",
        "-nostats",
    ]

    if use_progress:
        cmd += [
            "-progress",
            "pipe:2",
        ]

    # This must appear before -i because it is a decoder option.
    # It tells FFmpeg to output only keyframes during the analysis pass.
    if keyframes_only:
        cmd += [
            "-skip_frame",
            "nokey",
        ]

    if use_cuda:
        cmd += [
            "-hwaccel",
            "cuda",
        ]

    if keyframes_only:
        # Do not apply fps= here. The whole point is to analyse the native
        # keyframe stream with as little decoding/filtering work as possible.
        pix_fmt = "rgb24" if color_detect else "gray"
        filters = []
        if use_showinfo:
            filters.append("showinfo")
        filters.extend([f"scale={width}:{height}:flags=fast_bilinear", f"format={pix_fmt}"])
        vf = ",".join(filters)
    else:
        pix_fmt = "rgb24" if color_detect else "gray"
        vf = f"fps={sample_fps},scale={width}:{height}:flags=fast_bilinear,format={pix_fmt}"

    cmd += [
        "-i",
        str(input_path),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        vf,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24" if color_detect else "gray",
        "-",
    ]

    return cmd


def summarize_keyframe_spacing(keyframe_timestamps: list[float]) -> str | None:
    if len(keyframe_timestamps) < 2:
        return None

    gaps = [
        later - earlier
        for earlier, later in zip(keyframe_timestamps, keyframe_timestamps[1:])
        if later >= earlier
    ]

    if not gaps:
        return None

    avg_gap = sum(gaps) / len(gaps)
    min_gap = min(gaps)
    max_gap = max(gaps)

    if max_gap - min_gap <= 0.05:
        return f"about every {fmt_time(avg_gap)}"

    return f"avg {fmt_time(avg_gap)} (min {fmt_time(min_gap)}, max {fmt_time(max_gap)})"


def summarize_live_keyframe_spacing(keyframe_timestamps: list[float]) -> str:
    gaps = [
        later - earlier
        for earlier, later in zip(keyframe_timestamps, keyframe_timestamps[1:])
        if later >= earlier
    ]

    if not gaps:
        return "kf_gap=measuring"

    avg_gap = sum(gaps) / len(gaps)
    max_gap = max(gaps)
    return f"kf_gap=avg {fmt_time(avg_gap)} max {fmt_time(max_gap)}"


def detect_motion(
    input_path: Path,
    info: VideoInfo,
    *,
    width: int,
    sample_fps: float,
    pixel_threshold: int,
    motion_threshold: int,
    min_consecutive: int,
    use_cuda: bool,
    keyframes_only: bool,
    color_detect: bool,
    timestamp_mode: str,
    debug_every: float,
    quiet: bool,
) -> list[MotionFrame]:
    height = safe_even_height(info.width, info.height, width)
    frame_bytes = width * height * (3 if color_detect else 1)

    if not quiet:
        print("Scan settings")
        print(f"  Input             : {input_path}")
        print(
            f"  Source            : {info.width}x{info.height}, {info.fps:.3f} fps")
        print(f"  Duration          : {fmt_time(info.duration)}")
        mode_label = "RGB" if color_detect else "gray"
        if keyframes_only:
            print(
                f"  Analysis          : {width}x{height}, keyframes only, {mode_label}")
        else:
            print(
                f"  Analysis          : {width}x{height}, {sample_fps:g} fps, {mode_label}")
        print(f"  CUDA decode        : {'enabled' if use_cuda else 'disabled'}")
        print(
            f"  Keyframes only     : {'enabled' if keyframes_only else 'disabled'}")
        if keyframes_only:
            print(f"  Timestamp mode    : {timestamp_mode}")
            if timestamp_mode == "exact":
                print("  Keyframe spacing  : measuring during scan")
        print(f"  Pixel threshold   : {pixel_threshold}")
        print(f"  Motion threshold  : {motion_threshold} changed pixels")
        print(f"  Min consecutive   : {min_consecutive}")
        print()

    cmd = build_decode_command(
        input_path=input_path,
        width=width,
        height=height,
        sample_fps=sample_fps,
        use_cuda=use_cuda,
        keyframes_only=keyframes_only,
        color_detect=color_detect,
        timestamp_mode=timestamp_mode,
    )

    keyframe_timestamps: list[float] = []
    collect_keyframe_timestamps = keyframes_only and timestamp_mode == "exact"
    collect_progress = keyframes_only and timestamp_mode == "approx"
    timestamp_queue: queue.Queue[float] = queue.Queue()
    progress_queue: queue.Queue[float] = queue.Queue()
    stderr_lines: list[str] = []
    stderr_thread: threading.Thread | None = None

    pipe = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes * 32,
    )

    if pipe.stdout is None:
        die("Could not read FFmpeg stdout")

    if pipe.stderr is None:
        die("Could not read FFmpeg stderr")
    stderr_thread = threading.Thread(
        target=read_showinfo_stderr,
        args=(pipe.stderr, timestamp_queue, progress_queue, stderr_lines),
        daemon=True,
    )
    stderr_thread.start()

    previous_i16: np.ndarray | None = None
    current_i16: np.ndarray | None = None
    diff_buffer: np.ndarray | None = None
    motion_frames: list[MotionFrame] = []

    frame_index = 0
    consecutive_motion = 0
    pending_motion: list[MotionFrame] = []

    start_wall = time.time()
    last_debug = start_wall
    last_changed = 0
    peak_changed = 0
    last_progress_s = 0.0

    while True:
        raw = pipe.stdout.read(frame_bytes)

        if len(raw) == 0:
            break

        if len(raw) != frame_bytes:
            print()
            print(
                f"WARNING: short frame read: {len(raw)} bytes, expected {frame_bytes}")
            break

        if color_detect:
            current = np.frombuffer(
                raw, dtype=np.uint8).reshape((height, width, 3))
        else:
            current = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))

        changed = 0

        if current_i16 is None:
            current_i16 = np.empty_like(current, dtype=np.int16)
        np.copyto(current_i16, current, casting="unsafe")

        if previous_i16 is not None:
            if diff_buffer is None:
                diff_buffer = np.empty_like(current, dtype=np.int16)
            np.subtract(current_i16, previous_i16, out=diff_buffer)
            np.abs(diff_buffer, out=diff_buffer)
            if color_detect:
                changed = int(np.count_nonzero(np.any(diff_buffer > pixel_threshold, axis=2)))
            else:
                changed = int(np.count_nonzero(diff_buffer > pixel_threshold))
            last_changed = changed
            peak_changed = max(peak_changed, changed)

            if keyframes_only:
                t = float(frame_index)
            else:
                t = frame_index / sample_fps

            if changed >= motion_threshold:
                mf = MotionFrame(time_s=t, changed_pixels=changed)
                pending_motion.append(mf)
                consecutive_motion += 1

                if consecutive_motion >= min_consecutive:
                    motion_frames.extend(pending_motion)
                    pending_motion.clear()
            else:
                consecutive_motion = 0
                pending_motion.clear()

        if previous_i16 is None:
            previous_i16 = current_i16.copy()
        else:
            previous_i16, current_i16 = current_i16, previous_i16
        frame_index += 1

        now = time.time()
        if not quiet and now - last_debug >= debug_every:
            elapsed = now - start_wall

            if keyframes_only and not collect_keyframe_timestamps:
                last_progress_s = drain_progress_queue(progress_queue, last_progress_s)
                if last_progress_s > 0:
                    processed_s = min(info.duration, last_progress_s)
                    speed = processed_s / elapsed if elapsed > 0 else 0
                    pct = min(100.0, (processed_s / info.duration)
                              * 100.0) if info.duration > 0 else 0
                    remaining_s = max(0.0, info.duration - processed_s)
                    eta_s = remaining_s / speed if speed > 0 else 0.0
                    print(
                        "\r"
                        f"Scanned {fmt_time(processed_s)} / {fmt_time(info.duration)} "
                        f"({pct:5.1f}%)  "
                        f"ETA {fmt_time(eta_s)}  "
                        f"{speed:6.1f}x realtime  "
                        f"keyframes={frame_index:,}  "
                        f"motion={len(motion_frames):,}  "
                        f"changed={last_changed:,}  "
                        f"peak={peak_changed:,}",
                        end="",
                        flush=True,
                    )
                else:
                    keyframe_rate = frame_index / elapsed if elapsed > 0 else 0.0
                    print(
                        "\r"
                        f"Scanned keyframes={frame_index:,}  "
                        f"elapsed={fmt_time(elapsed)}  "
                        f"{keyframe_rate:6.1f} keyframes/s  "
                        f"motion={len(motion_frames):,}  "
                        f"changed={last_changed:,}  "
                        f"peak={peak_changed:,}  "
                        "waiting for FFmpeg progress",
                        end="",
                        flush=True,
                    )
                last_debug = now
                continue

            if keyframes_only:
                if collect_keyframe_timestamps:
                    drain_timestamp_queue(timestamp_queue, keyframe_timestamps)
                processed_s = estimate_keyframe_time(frame_index, keyframe_timestamps)
            else:
                processed_s = frame_index / sample_fps

            speed = processed_s / elapsed if elapsed > 0 else 0
            pct = min(100.0, (processed_s / info.duration)
                      * 100.0) if info.duration > 0 else 0
            remaining_s = max(0.0, info.duration - processed_s)
            eta_s = remaining_s / speed if speed > 0 else 0.0

            print(
                "\r"
                f"Scanned {fmt_time(processed_s)} / {fmt_time(info.duration)} "
                f"({pct:5.1f}%)  "
                f"ETA {fmt_time(eta_s)}  "
                f"{speed:6.1f}x realtime  "
                f"frames={frame_index:,}  "
                f"motion={len(motion_frames):,}  "
                f"changed={last_changed:,}  "
                f"peak={peak_changed:,}"
                + (
                    f"  {summarize_live_keyframe_spacing(keyframe_timestamps)}"
                    if collect_keyframe_timestamps
                    else ""
                ),
                end="",
                flush=True,
            )

            last_debug = now

    stderr_output = b""

    try:
        pipe.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pipe.kill()
        pipe.wait()

    if stderr_thread is not None:
        stderr_thread.join()
    drain_timestamp_queue(timestamp_queue, keyframe_timestamps)
    last_progress_s = drain_progress_queue(progress_queue, last_progress_s)
    stderr_output = "\n".join(stderr_lines).encode()

    if pipe.returncode not in (0, None):
        print()
        print("FFmpeg decode failed.")
        if stderr_output:
            print(stderr_output.decode(errors="replace"))
        die("FFmpeg exited with an error")

    if keyframes_only:
        if len(keyframe_timestamps) < frame_index:
            if collect_keyframe_timestamps:
                print()
                if keyframe_timestamps:
                    print(
                        "WARNING: incomplete keyframe timestamps captured; "
                        "estimating some event times"
                    )
                else:
                    print("WARNING: no keyframe timestamps captured; estimating event times")
        motion_frames = map_keyframe_motion_times(
            motion_frames,
            keyframe_timestamps=keyframe_timestamps,
            decoded_keyframes=frame_index,
            duration=info.duration,
        )

    scan_elapsed = time.time() - start_wall

    if not quiet:
        print()
    if keyframes_only and collect_keyframe_timestamps:
        spacing_summary = summarize_keyframe_spacing(keyframe_timestamps)
        if spacing_summary is not None:
            print(f"Observed keyframe spacing: {spacing_summary}")
            print()
    print(f"Scan complete. Motion frames detected: {len(motion_frames):,}")
    if info.duration > 0 and scan_elapsed > 0:
        print(f"Scan speed: {info.duration / scan_elapsed:,.1f}x realtime")
    print()

    return motion_frames


