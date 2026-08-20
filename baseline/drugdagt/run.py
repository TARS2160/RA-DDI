from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = name.strip().strip('"').strip("'")
    name = name.replace("\\", "_").replace("/", "_").replace(":", "_")
    name = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "command"
    return name[:max_len]


def get_arg_value(command: str, arg_name: str) -> str:
    pattern = rf"{re.escape(arg_name)}\s+(\"[^\"]+\"|'[^']+'|\S+)"
    m = re.search(pattern, command)
    if not m:
        return ""
    return m.group(1).strip().strip('"').strip("'")


def infer_command_name(command: str, index: int) -> str:
    save_dir = get_arg_value(command, "--save_dir")
    if save_dir:
        name = Path(save_dir.replace("\\", "/")).name
        return sanitize_filename(f"{index:03d}_{name}")

    data_path = get_arg_value(command, "--data_path")
    if data_path:
        # strict_data/pdd_graph_2class_s42_f3/train_pair_left.csv -> pdd_graph_2class_s42_f3
        parts = Path(data_path.replace("\\", "/")).parts
        if len(parts) >= 2:
            return sanitize_filename(f"{index:03d}_{parts[-2]}")

    return sanitize_filename(f"{index:03d}_command")


def load_commands(run_file: Path) -> Tuple[List[Dict[str, str]], int, int]:
    active: List[Dict[str, str]] = []
    empty_count = 0
    comment_count = 0

    with run_file.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                empty_count += 1
                continue
            if line.startswith("#"):
                comment_count += 1
                continue
            active.append({"line_no": str(line_no), "command": line})

    return active, empty_count, comment_count


def run_one_command(command: str, log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()

    with log_path.open("w", encoding="utf-8", errors="replace", newline="") as log:
        header = [
            "=" * 120,
            f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Working dir: {cwd}",
            "Command:",
            command,
            "=" * 120,
            "",
        ]
        for h in header:
            print(h)
            log.write(h + "\n")
        log.flush()

        if DRY_RUN:
            msg = "[DRY_RUN] Command was not executed.\n"
            print(msg)
            log.write(msg)
            return 0

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None

        try:
            for out_line in process.stdout:
                print(out_line, end="")
                log.write(out_line)
                log.flush()
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Terminating current command...")
            log.write("\n[INTERRUPTED] Terminating current command...\n")
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
            raise

        return_code = process.wait()
        elapsed = time.time() - start

        footer = [
            "",
            "=" * 120,
            f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Elapsed seconds: {elapsed:.2f}",
            f"Return code: {return_code}",
            "=" * 120,
            "",
        ]
        for h in footer:
            print(h)
            log.write(h + "\n")
        log.flush()

        return return_code


def write_summary_csv(summary_path: Path, rows: List[Dict[str, str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "line_no",
        "name",
        "status",
        "return_code",
        "elapsed_seconds",
        "log_path",
        "warnings",
        "command",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main() -> int:
    script_dir = Path(__file__).resolve().parent

    run_file = script_dir / RUN_FILE_NAME
    if not run_file.exists():
        alt = script_dir / "run(1).txt"
        if alt.exists():
            run_file = alt
        else:
            print(f"[ERROR] Cannot find {RUN_FILE_NAME} or run(1).txt in: {script_dir}")
            return 1

    timestamp = now_str()
    log_dir = script_dir / LOG_ROOT / timestamp
    summary_dir = script_dir / SUMMARY_DIR
    summary_csv = summary_dir / f"batch_from_run_txt_summary_{timestamp}.csv"
    latest_summary_csv = summary_dir / "batch_from_run_txt_summary_latest.csv"

    commands, empty_count, comment_count = load_commands(run_file)

    print("=" * 120)
    print("DrugDAGT batch runner")
    print(f"Script dir: {script_dir}")
    print(f"Run file: {run_file}")
    print(f"Log dir: {log_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Active commands: {len(commands)}")
    print(f"Skipped empty lines: {empty_count}")
    print(f"Skipped comment lines: {comment_count}")
    print("=" * 120)

    if not commands:
        print("[ERROR] No active commands found.")
        return 1

    results: List[Dict[str, str]] = []
    failed_rows: List[Dict[str, str]] = []

    for idx, item in enumerate(commands, start=1):
        line_no = item["line_no"]
        command = item["command"]
        name = infer_command_name(command, idx)
        log_path = log_dir / f"{name}.log"
        warnings = check_command_warnings(command)

        print()
        print("=" * 120)
        print(f"Running {idx}/{len(commands)} | line {line_no} | {name}")
        if warnings:
            print("[WARNINGS]")
            for w in warnings:
                print(f"  - {w}")
        print(f"Log: {log_path}")
        print("=" * 120)

        start = time.time()
        status = "UNKNOWN"
        return_code = -999

        try:
            return_code = run_one_command(command, log_path, script_dir)
            elapsed = time.time() - start

            if return_code == 0:
                status = "SUCCESS"
            else:
                status = "FAILED_SKIPPED"
                print(f"[FAILED] Command {idx} failed with return code {return_code}. Continue to next command.")
        except KeyboardInterrupt:
            elapsed = time.time() - start
            status = "INTERRUPTED"
            return_code = -130
            print("[INTERRUPTED] Batch run interrupted by user.")
        except Exception as e:
            elapsed = time.time() - start
            status = "FAILED_SKIPPED"
            return_code = -1
            print(f"[EXCEPTION] Command {idx} raised exception: {repr(e)}. Continue to next command.")

            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", errors="replace") as log:
                log.write("\n" + "=" * 120 + "\n")
                log.write(f"[EXCEPTION] {repr(e)}\n")
                log.write("=" * 120 + "\n")

        row = {
            "index": str(idx),
            "line_no": line_no,
            "name": name,
            "status": status,
            "return_code": str(return_code),
            "elapsed_seconds": f"{elapsed:.2f}",
            "log_path": str(log_path),
            "warnings": " | ".join(warnings),
            "command": command,
        }
        results.append(row)

        if status != "SUCCESS":
            failed_rows.append(row)

        write_summary_csv(summary_csv, results)
        write_summary_csv(latest_summary_csv, results)

        if status == "INTERRUPTED":
            break

        if status != "SUCCESS" and STOP_ON_FAILURE:
            break

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results if r["status"] != "SUCCESS")
    not_run_count = len(commands) - len(results)

    print()
    print("=" * 120)
    print("Batch finished")
    print(f"Active commands total: {len(commands)}")
    print(f"Success: {success_count}")
    print(f"Failed / skipped after failure / interrupted: {failed_count}")
    print(f"Not run because batch stopped/interrupted: {not_run_count}")
    print(f"Skipped empty lines: {empty_count}")
    print(f"Skipped comment lines: {comment_count}")
    print(f"Log dir: {log_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Latest summary CSV: {latest_summary_csv}")

    if failed_rows:
        print()
        print("Failed / skipped commands:")
        for r in failed_rows:
            print(
                f"  - index={r['index']}, line={r['line_no']}, "
                f"name={r['name']}, return_code={r['return_code']}, log={r['log_path']}"
            )
    else:
        print()
        print("No failed commands.")

    if not_run_count > 0:
        print()
        print("Commands not run:")
        for item in commands[len(results):]:
            print(f"  - line={item['line_no']}: {item['command'][:200]}...")

    print("=" * 120)

    return 0


if __name__ == "__main__":
    sys.exit(main())
