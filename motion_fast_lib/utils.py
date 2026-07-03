from __future__ import annotations

import subprocess
import sys


def die(message: str, exit_code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def run_command(
    cmd: list[str],
    *,
    capture_output: bool = False,
    check: bool = True,
    input_data: str | None = None,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check,
            input=input_data,
        )
    except FileNotFoundError:
        die(f"Command not found: {cmd[0]}")
    except subprocess.CalledProcessError as exc:
        print()
        print("Command failed:")
        print(" ".join(cmd))
        if exc.stdout:
            print()
            print("STDOUT:")
            print(exc.stdout)
        if exc.stderr:
            print()
            print("STDERR:")
            print(exc.stderr)
        die(f"{cmd[0]} exited with code {exc.returncode}")


def parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None

    if "/" in value:
        num_s, den_s = value.split("/", 1)
        num = float(num_s)
        den = float(den_s)
        if den == 0:
            return None
        return num / den

    return float(value)


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))

    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60

    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def safe_even_height(src_w: int, src_h: int, dst_w: int) -> int:
    raw_h = src_h * dst_w / src_w
    h = max(2, int(round(raw_h)))

    if h % 2:
        h += 1

    return h
