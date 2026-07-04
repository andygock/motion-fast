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


def read_showinfo_stderr(
    stderr_pipe,
    timestamp_queue: queue.Queue[float],
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
        elif line:
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
) -> list[str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info" if keyframes_only else "error",
        "-nostats",
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
        vf = f"showinfo,scale={width}:{height}:flags=fast_bilinear,format={pix_fmt}"
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
    debug_every: float,
) -> list[MotionFrame]:
    height = safe_even_height(info.width, info.height, width)
    frame_bytes = width * height * (3 if color_detect else 1)

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
    )

    keyframe_timestamps: list[float] = []
    timestamp_queue: queue.Queue[float] = queue.Queue()
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

    if keyframes_only:
        if pipe.stderr is None:
            die("Could not read FFmpeg stderr for keyframe timestamps")
        stderr_thread = threading.Thread(
            target=read_showinfo_stderr,
            args=(pipe.stderr, timestamp_queue, stderr_lines),
            daemon=True,
        )
        stderr_thread.start()

    previous: np.ndarray | None = None
    motion_frames: list[MotionFrame] = []

    frame_index = 0
    consecutive_motion = 0
    pending_motion: list[MotionFrame] = []

    start_wall = time.time()
    last_debug = start_wall
    last_changed = 0
    peak_changed = 0

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

        if previous is not None:
            diff = np.abs(current.astype(np.int16) - previous.astype(np.int16))
            if color_detect:
                changed = int(np.count_nonzero(np.any(diff > pixel_threshold, axis=2)))
            else:
                changed = int(np.count_nonzero(diff > pixel_threshold))
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

        previous = current.copy()
        frame_index += 1

        now = time.time()
        if now - last_debug >= debug_every:
            if keyframes_only:
                drain_timestamp_queue(timestamp_queue, keyframe_timestamps)
                processed_s = estimate_keyframe_time(frame_index, keyframe_timestamps)
            else:
                processed_s = frame_index / sample_fps

            elapsed = now - start_wall
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
                    if keyframes_only
                    else ""
                ),
                end="",
                flush=True,
            )

            last_debug = now

    stderr_output = b""

    if keyframes_only:
        try:
            pipe.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pipe.kill()
            pipe.wait()

        if stderr_thread is not None:
            stderr_thread.join()
        drain_timestamp_queue(timestamp_queue, keyframe_timestamps)
        stderr_output = "\n".join(stderr_lines).encode()
    else:
        try:
            _, stderr_output = pipe.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pipe.kill()
            _, stderr_output = pipe.communicate()

    if pipe.returncode not in (0, None):
        print()
        print("FFmpeg decode failed.")
        if stderr_output:
            print(stderr_output.decode(errors="replace"))
        die("FFmpeg exited with an error")

    if keyframes_only:
        if len(keyframe_timestamps) < frame_index:
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

    print()
    if keyframes_only:
        spacing_summary = summarize_keyframe_spacing(keyframe_timestamps)
        if spacing_summary is not None:
            print(f"Observed keyframe spacing: {spacing_summary}")
            print()
    print(f"Scan complete. Motion frames detected: {len(motion_frames):,}")
    print()

    return motion_frames


