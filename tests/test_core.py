from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from motion_fast_lib.analysis import map_keyframe_motion_times
from motion_fast_lib.models import MotionFrame
from motion_fast_lib.runner import output_dir_for_input, prepare_output_dir
from motion_fast_lib.utils import fmt_time


# These tests cover pure helper behavior and filesystem safety checks. They do
# not invoke FFmpeg, so they are fast enough to run frequently while still
# protecting the edge cases that caused the robustness fixes.
class UtilityTests(unittest.TestCase):
    def test_fmt_time_carries_rounded_milliseconds(self) -> None:
        # fmt_time rounds to the nearest millisecond. This value rounds up from
        # 1.9996s to exactly 2.000s, so the millisecond carry must increment the
        # seconds field instead of producing an invalid ".1000" suffix.
        self.assertEqual(fmt_time(1.9996), "00:00:02.000")

    def test_fmt_time_carries_to_next_minute(self) -> None:
        # Same rounding edge as above, but across a minute boundary. This makes
        # sure the carry propagates through seconds into minutes.
        self.assertEqual(fmt_time(59.9996), "00:01:00.000")


class KeyframeTimestampTests(unittest.TestCase):
    def test_maps_motion_frame_ordinals_to_keyframe_timestamps(self) -> None:
        # During keyframe-only scanning, motion detections are first stored with
        # their decoded keyframe ordinal in MotionFrame.time_s. After FFmpeg has
        # exited, map_keyframe_motion_times replaces those ordinals with the
        # real showinfo timestamps collected from stderr.
        frames = [
            MotionFrame(time_s=1.0, changed_pixels=100),
            MotionFrame(time_s=3.0, changed_pixels=200),
        ]

        mapped = map_keyframe_motion_times(
            frames,
            keyframe_timestamps=[0.0, 5.0, 11.0, 23.0],
            decoded_keyframes=4,
            duration=30.0,
        )

        # Ordinal 1 maps to timestamp 5.0, ordinal 3 maps to timestamp 23.0.
        # changed_pixels is detection metadata and should pass through exactly.
        self.assertEqual([frame.time_s for frame in mapped], [5.0, 23.0])
        self.assertEqual([frame.changed_pixels for frame in mapped], [100, 200])

    def test_scales_ordinals_when_no_timestamps_are_available(self) -> None:
        # If FFmpeg does not provide any showinfo timestamps, the code falls
        # back to spreading decoded keyframe ordinals across the known video
        # duration. With 5 decoded keyframes over 20 seconds, ordinal 2 lands at
        # 10 seconds: 20 / (5 - 1) * 2.
        frames = [MotionFrame(time_s=2.0, changed_pixels=100)]

        mapped = map_keyframe_motion_times(
            frames,
            keyframe_timestamps=[],
            decoded_keyframes=5,
            duration=20.0,
        )

        self.assertEqual(mapped[0].time_s, 10.0)


class OutputDirectoryTests(unittest.TestCase):
    def test_prepare_output_dir_keeps_unrelated_files(self) -> None:
        # prepare_output_dir used to remove an existing output directory before
        # writing events.csv. The current behavior is intentionally safer: leave
        # the directory in place and only allow the generated events.csv file to
        # be overwritten by the later write step.
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            unrelated = output_dir / "notes.txt"
            unrelated.write_text("keep me", encoding="utf-8")

            prepare_output_dir(output_dir, keep_existing=False)

            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep me")

    def test_prepare_output_dir_rejects_file_path(self) -> None:
        # A user can accidentally pass --out-dir pointing at a file. The runner
        # should fail with the project's normal SystemExit path instead of
        # raising a raw filesystem exception or traceback.
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "not-a-dir"
            output_path.write_text("", encoding="utf-8")

            # die() writes the user-facing error to stderr. Redirect it here so
            # the expected failure path does not pollute successful test output.
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    prepare_output_dir(output_path, keep_existing=False)

    def test_multiple_inputs_use_per_input_subdirectories(self) -> None:
        # For multiple input videos, a single --out-dir acts as a parent folder.
        # Each input gets its own generated subdirectory so their events.csv
        # files cannot overwrite each other.
        args = argparse.Namespace(out_dir=Path("reviews"))
        output_dir = output_dir_for_input(Path("camera.avi"), args, input_count=2)

        self.assertEqual(output_dir, Path("reviews").resolve() / "camera_motion_review")


if __name__ == "__main__":
    unittest.main()
