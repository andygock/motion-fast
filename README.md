# motion-fast

Fast CCTV motion review generator for one or more video files.

`motion-fast` scans input videos for motion, merges nearby detections into events, and builds condensed review MP4 files directly from the source segments. By default it scans only video keyframes for faster analysis; use `--all-frames` for the slower sampled-frame scan when you need to catch motion between keyframes.

The project can be installed as the `motion-fast` command, or run directly with `python -m motion_fast ...`. The implementation is split into the `motion_fast_lib/` package:

- `analysis.py`: FFmpeg decode command construction and NumPy motion scanning.
- `events.py`: motion-frame merging and `events.csv` writing.
- `review.py`: in-memory FFmpeg concat-list and review video creation.
- `runner.py` and `cli.py`: input resolution, output handling, and CLI orchestration.

The review stage is a single FFmpeg pass. The concat list is piped to FFmpeg in memory, so normal review runs do not create a temporary output directory or sidecar concat file. In re-encoded mode, `--extract-workers` controls FFmpeg thread count and auto-selects all logical CPUs.

## Requirements

- Python 3.10 or newer
- FFmpeg and FFprobe available on `PATH`
- NumPy
- Optional: NVIDIA CUDA/NVENC support in FFmpeg for faster decode and encode

## Installation

For development from this checkout:

    python -m pip install -e .

For editable development from this checkout when not using a virtual environment:

    python -m pip install --user -e .

For a normal install for the current user only:

    python -m pip install --user .

For a system-wide install, run from an elevated terminal:

    python -m pip install .

To install directly from a GitHub repo:

    python -m pip install git+https://github.com/andygock/motion-fast

Or, if you want an editable checkout after cloning:

    git clone https://github.com/andygock/motion-fast
    cd motion-fast
    python -m pip install -e .

After installation, run the CLI as:

    motion-fast INPUT.avi

You can also run the installed Python module form:

    python -m motion_fast INPUT.avi

The package/distribution name and installed command use a dash: `motion-fast`. The Python module uses an underscore: `motion_fast`, because Python module names cannot contain dashes.

If you do not install the project, install the Python dependency manually:

    python -m pip install numpy

Check FFmpeg availability:

    ffmpeg -version
    ffprobe -version

## Basic Usage

Single file:

    motion-fast INPUT.avi

Glob pattern:

    motion-fast /path/*.avi

Without installing the project, use:

    python -m motion_fast INPUT.avi

Multiple files or glob matches are processed one at a time. All CLI switches are applied to each input file independently.

By default, each input creates only:

- `review_<input_stem>.mp4` beside the input video

No `<input_stem>_motion_review` directory is created during a normal review run. Use `--write-events-csv` if you also want an `events.csv` sidecar, or `--detect-only` if you only want the event log.

Example:

    motion-fast INPUT.avi

For an input named `INPUT.avi`, the final review video is `review_INPUT.mp4`

## Common Commands

Normal review, default settings. This scans keyframes only:

    motion-fast INPUT.avi

Keyframe-only analysis can be 10x faster or more. But if keyframes are too far apart, it may miss events. `--keyframes-only` is accepted for explicitness, but it is already the default:

    motion-fast INPUT.avi --keyframes-only

Use sampled frames instead of only keyframes when keyframe spacing is too wide or motion is being missed:

    motion-fast INPUT.avi --all-frames

Scan first 60s of video and check keyframe spacing first with:

    ffprobe -v error -read_intervals "%+60" -select_streams v:0 -skip_frame nokey -show_entries frame=best_effort_timestamp_time -of csv=p=0 INPUT.avi

Fast review with GPU encoding and 32x playback:

    motion-fast INPUT.avi --speed 32 --nvenc

Lower-resolution, low-FPS sampled-frame scan with wider event padding:

    motion-fast INPUT.avi --all-frames --width 160 --fps 0.5 --motion-threshold 400 --pixel-threshold 35 --merge-gap 20 --pre-roll 5 --post-roll 8 --speed 16 --nvenc

Detect motion and write `events.csv` only:

    motion-fast /path/*.avi --detect-only

Build a review and also keep the event log:

    motion-fast /path/*.avi --write-events-csv

Very fast detect-only scan:

    motion-fast /path/*.avi --width 160 --motion-threshold 400 --pixel-threshold 35 --merge-gap 30 --pre-roll 8 --post-roll 12 --detect-only

In keyframe-only mode, the script no longer runs a separate `ffprobe` keyframe timestamp pass. It reads FFmpeg `showinfo` timestamps while scanning the keyframes; if timestamps are unavailable, motion times are estimated across the full video duration.

## Tuning Motion Detection

Useful options:

- `--width`: Analysis width in pixels. Smaller is faster but less detailed. Default: `320`.
- `--fps`: Analysis sample rate for `--all-frames` mode. Lower values are faster but can miss short motion. Default: `2`.
- `--pixel-threshold`: Per-pixel brightness difference required to count as changed. Default: `30`.
- `--motion-threshold`: Number of changed pixels required to mark a frame as motion. Default: `200`.
- `--min-consecutive`: Consecutive motion frames required before motion is accepted. Default: `1`.
- `--merge-gap`: Merge detections separated by this many seconds or less. Default: `6`.
- `--pre-roll`: Seconds to include before detected motion. Default: `0.5`.
- `--post-roll`: Seconds to include after detected motion. Default: `0.5`.
- `--min-event-duration`: Discard merged events shorter than this many seconds. Default: `1`.

If too much footage is included, increase `--motion-threshold` or `--pixel-threshold`.

If motion is missed, decrease `--motion-threshold`, decrease `--pixel-threshold`, use `--color-detect`, or switch to `--all-frames`. In `--all-frames` mode, increasing `--fps` can also help catch shorter motion.

## Output Options

- `--out-dir PATH`: Directory for `events.csv` when it is written. With multiple inputs, this is used as a parent directory and each input gets its own `<input_stem>_motion_review` subdirectory.
- `--keep-existing`: Do not delete an existing `events.csv` output directory before running.
- `--no-clobber`: Skip processing if the final review MP4 already exists.
- `--detect-only`: Only scan and write `events.csv`; do not build a review video.
- `--write-events-csv`: Write `events.csv` during normal review runs. Detect-only mode always writes it.
- `--keyframes-only`: Scan only video keyframes. Default: enabled.
- `--all-frames`: Scan sampled frames instead of only keyframes. Slower, but can catch motion between keyframes. The sample rate is controlled by `--fps`.
- `--speed N`: Speed up the review video by `N`. Default: `1`.
- `--copy-video`: Stream-copy the review. Default: enabled. Fastest, but cuts may be less accurate.
- `--reencode-clips`: Re-encode the review instead of stream-copying it.
- `--extract-workers N`: FFmpeg thread count used when encoding the review. Default: `0` = auto, using all logical CPUs.

## GPU Options

CUDA decode is enabled by default during scanning. Disable it with:

    motion-fast input.avi --no-cuda-decode

Use NVIDIA NVENC for review encoding:

    motion-fast input.avi --nvenc

Fastest review generation is usually:

    motion-fast input.avi

When you need speed-up or re-encoding, use multi-threaded encoding:

    motion-fast input.avi --reencode-clips --speed 4 --extract-workers 8

If FFmpeg was not built with CUDA or NVENC support, use `--no-cuda-decode` and omit `--nvenc`.

## Notes

- The script accepts one or more input video files or glob patterns, such as `*.avi` or `/path/*.avi`.
- Inaccessible files matched by a glob pattern are reported and skipped. If every match is inaccessible, the run exits with `No accessible input files were found.`
- It was designed for Windows CCTV AVI/H.264 footage, but should work with other FFmpeg-readable video files.
- Keyframe-only mode is the default. It avoids a separate keyframe timestamp probe and is faster, but can miss motion that occurs between keyframes.
- In keyframe-only mode, the scan stats report the observed average keyframe spacing once enough timestamps are collected, without an extra probe pass.
- The final review MP4 is written beside each input video, not inside the output directory.
- Normal review runs keep the event list and FFmpeg concat manifest in memory. `events.csv` is written only with `--detect-only` or `--write-events-csv`.
