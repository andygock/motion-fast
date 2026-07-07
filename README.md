# motion-fast

Fast CCTV motion review generator for one or more video files.

`motion-fast` scans input videos for motion, merges nearby detections into events, and builds condensed review MP4 files directly from the source segments. By default it scans only video keyframes for faster analysis; use `--all-frames` for the slower sampled-frame scan when you need to catch motion between keyframes.

The project can be installed as the `motion-fast` command, or run directly with `python -m motion_fast ...`. The implementation is split into the `motion_fast_lib/` package:

- `analysis.py`: FFmpeg decode command construction and NumPy motion scanning.
- `events.py`: motion-frame merging and `events.csv` writing.
- `review.py`: in-memory FFmpeg concat-list and review video creation.
- `runner.py` and `cli.py`: input resolution, output handling, and CLI orchestration.

The review stage is a single FFmpeg pass. The concat list is piped to FFmpeg in memory, so normal review runs do not create a temporary output directory or sidecar concat file. In re-encoded mode, `--extract-workers` controls FFmpeg thread count and auto-selects all logical CPUs.
When re-encoding is required, `--encoder auto` uses NVENC if FFmpeg can run `h264_nvenc`, otherwise it falls back to libx264.

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
Existing review MP4 files are not overwritten by default. Use `--overwrite` to replace an existing review file.
Before scanning starts, the command prints an output preflight check showing the number of existing output files as of now, split into files that will not be overwritten and files that will be overwritten. Each input still checks the file state again when it actually starts, so later filesystem changes are handled according to the state at processing time.

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

Overwrite an existing review MP4:

    motion-fast INPUT.avi --overwrite

Very fast detect-only scan:

    motion-fast /path/*.avi --width 160 --motion-threshold 400 --pixel-threshold 35 --merge-gap 30 --pre-roll 8 --post-roll 12 --detect-only

Maximum-throughput keyframe scan with approximate timestamps and no live progress:

    motion-fast /path/*.avi --timestamp-mode approx --quiet

Process several input files in parallel:

    motion-fast /path/*.avi --jobs 4 --quiet

In keyframe-only mode, the script no longer runs a separate `ffprobe` keyframe timestamp pass. It reads FFmpeg `showinfo` timestamps while scanning the keyframes, shows the observed average and maximum keyframe gap in the live progress status, then maps detected keyframe ordinals to the complete timestamp list after the scan. If timestamps are unavailable or incomplete, motion times are estimated and the script prints a warning.

For maximum scan speed, use `--timestamp-mode approx`. This skips FFmpeg `showinfo` logging and maps decoded keyframe ordinals across the known video duration. It is faster and quieter, but event times are approximate if keyframes are not evenly spaced.

## Maximum Speed Examples

Fastest normal review for one file:

    motion-fast INPUT.avi --timestamp-mode approx --quiet

Fast batch detect-only scan for several files:

    motion-fast /path/*.avi --detect-only --timestamp-mode approx --quiet --jobs 4 --width 160

Fast re-encoded review when you need playback speed-up:

    motion-fast INPUT.avi --speed 16 --encoder auto --timestamp-mode approx --quiet

Main tradeoffs:

- `--timestamp-mode approx` removes FFmpeg `showinfo` timestamp logging, but event times are estimated if keyframes are unevenly spaced.
- In `--timestamp-mode approx`, live progress uses FFmpeg's lightweight progress output for video position and speed. If FFmpeg has not emitted progress yet, the status temporarily falls back to keyframes per second until the first progress update arrives.
- `--jobs N` improves batch wall-clock time, but too many jobs can saturate disk, CPU, or GPU decode.
- Lower `--width` reduces analysis work, but can miss small or subtle motion.
- `--speed N` requires re-encoding, so use `--encoder auto` or `--encoder nvenc` when a GPU encoder is available.

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
- `--color-detect`: Compare RGB channels instead of grayscale so color-only changes can count as motion.

If too much footage is included, increase `--motion-threshold` or `--pixel-threshold`.

If motion is missed, decrease `--motion-threshold`, decrease `--pixel-threshold`, use `--color-detect`, or switch to `--all-frames`. In `--all-frames` mode, increasing `--fps` can also help catch shorter motion.

## Output Options

Before processing any input, `motion-fast` reports how many target output files already exist as of that moment. The report separates files that will not be overwritten from files that will be overwritten. This is only an initial status snapshot; per-input processing still uses the file state at the moment that input starts.

- `--out-dir PATH`: Directory for `events.csv` when it is written. With multiple inputs, this is used as a parent directory and each input gets its own `<input_stem>_<hash>_motion_review` subdirectory. The hash is derived from the resolved input path so same-named videos from different folders cannot overwrite each other's event logs. Existing unrelated files in this directory are left untouched.
- `--keep-existing`: Compatibility option. Event output directories are never deleted; existing unrelated files are kept.
- `--no-clobber`: Skip processing if the final review MP4 already exists. Default: enabled.
- `--overwrite`: Replace an existing final review MP4.
- `--detect-only`: Only scan and write `events.csv`; do not build a review video.
- `--write-events-csv`: Write `events.csv` during normal review runs. Detect-only mode always writes it.
- `--keyframes-only`: Scan only video keyframes. Default: enabled.
- `--all-frames`: Scan sampled frames instead of only keyframes. Slower, but can catch motion between keyframes. The sample rate is controlled by `--fps`.
- `--speed N`: Speed up the review video by `N`. Default: `1`.
- `--copy-video`: Stream-copy the review. Default: enabled. Fastest, but cuts may be less accurate.
- `--reencode-clips`: Re-encode the review instead of stream-copying it.
- `--extract-workers N`: FFmpeg thread count used when encoding the review. Default: `0` = auto, using all logical CPUs.
- `--crf N`: Quality value for encoded review output. libx264 uses CRF, NVENC uses CQ. Default: `28`.
- `--preset NAME`: libx264 preset when not using NVENC. Default: `veryfast`.
- `--debug-every N`: Progress update interval in seconds. Default: `1`.
- `--timestamp-mode exact|approx`: Keyframe timestamp mapping mode. `exact` parses FFmpeg `showinfo` timestamps. `approx` skips that logging and estimates keyframe times from video duration for more speed. Default: `exact`.
- `--quiet`: Suppress live scan progress. Final summaries are still printed.
- `--jobs N`: Process up to `N` input files in parallel. Default: `1`. Parallel jobs capture each file's log and print it when that file finishes.

## GPU Options

CUDA decode is enabled by default during scanning. Disable it with:

    motion-fast input.avi --no-cuda-decode

Use NVIDIA NVENC for review encoding:

    motion-fast input.avi --nvenc

Choose the encoder explicitly when a review must be re-encoded:

    motion-fast input.avi --speed 8 --encoder auto

Encoder choices are `auto`, `cpu`, and `nvenc`. `auto` is the default and uses NVENC when FFmpeg can run `h264_nvenc`; otherwise it uses libx264. `--nvenc` remains available as a shortcut for `--encoder nvenc`.

Fastest review generation is usually:

    motion-fast input.avi

When you need speed-up or re-encoding, use multi-threaded encoding:

    motion-fast input.avi --reencode-clips --speed 4 --extract-workers 8

If FFmpeg was not built with CUDA or NVENC support, use `--no-cuda-decode` and omit `--nvenc`.

## Tests

The test suite uses Python's built-in `unittest` runner and does not require FFmpeg. The current tests cover pure helper behavior and safety-sensitive edge cases:

- time formatting and millisecond carry behavior
- FFmpeg analysis command construction for exact and approximate keyframe timestamp modes
- live keyframe-spacing summary formatting
- keyframe ordinal-to-timestamp mapping and fallback estimation
- safe event output directory handling
- hashed per-input event directory naming for multi-input runs
- clean failure handling for malformed `ffprobe` metadata
- initial existing-output preflight overwrite classification
- per-input output directory naming for multiple inputs

Run the tests from the repository root:

    python -m unittest discover -s tests

You can also run a syntax/import check with:

    python -m compileall motion_fast.py motion_fast_lib tests

## Notes

- The script accepts one or more input video files or glob patterns, such as `*.avi` or `/path/*.avi`.
- Inaccessible files matched by a glob pattern are reported and skipped. If every match is inaccessible, the run exits with `No accessible input files were found.`
- It was designed for Windows CCTV AVI/H.264 footage, but should work with other FFmpeg-readable video files.
- Keyframe-only mode is the default. It avoids a separate keyframe timestamp probe and is faster, but can miss motion that occurs between keyframes.
- In keyframe-only mode, the live progress status includes `kf_gap=avg ... max ...` once enough timestamps are collected, without an extra probe pass. Large gaps mean motion between keyframes can be missed; cancel and rerun with `--all-frames` if the observed gap is too wide for your footage.
- The final review MP4 is written beside each input video, not inside the output directory.
- Normal review runs keep the event list and FFmpeg concat manifest in memory. `events.csv` is written only with `--detect-only` or `--write-events-csv`.
- Event output directories are created when needed but are not deleted. Writing `events.csv` overwrites that file only.
- Invalid numeric options, such as negative roll values or out-of-range threshold values, are rejected before scanning.
