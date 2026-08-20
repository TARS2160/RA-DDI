# -*- coding: utf-8 -*-
"""
Batch runner for DrugDAGT commands in run.txt.

功能：
1. 读取同目录下的 run.txt（如果没有，则尝试 run(1).txt）。
2. 跳过空行和以 # 开头的注释行。
3. 不自动删除 Tee-Object，不自动修正 split 路径，不自动修正 split_seed/split_fold。
4. 每条命令原样运行，并将终端输出单独保存为 log。
5. 某条命令失败时，不停止整个批处理，继续运行下一条。
6. 最后汇总成功、失败、未运行/跳过的命令，并保存 summary CSV。

使用方式：
- 将本脚本放到 DrugDAGT-main/code 目录。
- 将 run.txt 放到同一个目录。
- 在 PyCharm 中右键本脚本 -> Run。
"""

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


# =========================
# 用户可改配置
# =========================

# 只检查命令、不真正运行。正式跑请保持 False。
DRY_RUN = False

# 失败后是否继续下一条。你的需求是失败后跳过，所以保持 False。
STOP_ON_FAILURE = False

# 命令文件名。默认读取 run.txt；如果没有，会尝试 run(1).txt。
RUN_FILE_NAME = "run.txt"

# log 根目录，会自动创建。
LOG_ROOT = Path("strict_results") / "DrugDAGT" / "logs" / "batch_from_run_txt"

# summary 输出位置。
SUMMARY_DIR = Path("strict_results") / "DrugDAGT"


# =========================
# 工具函数
# =========================

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(name: str, max_len: int = 120) -> str:
    """将 save_dir/split 名转成安全文件名。"""
    name = name.strip().strip('"').strip("'")
    name = name.replace("\\", "_").replace("/", "_").replace(":", "_")
    name = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "command"
    return name[:max_len]


def get_arg_value(command: str, arg_name: str) -> str:
    """
    从命令字符串中读取参数值。
    支持：
      --save_dir strict_results/...
      --save_dir "strict_results/..."
    不修改命令本身。
    """
    pattern = rf"{re.escape(arg_name)}\s+(\"[^\"]+\"|'[^']+'|\S+)"
    m = re.search(pattern, command)
    if not m:
        return ""
    return m.group(1).strip().strip('"').strip("'")


def infer_command_name(command: str, index: int) -> str:
    """优先用 --save_dir 的最后一级目录作为 log 名；没有则用 --data_path 推断。"""
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
    """
    返回 active commands，同时统计空行和注释行。
    注释行定义：strip 后以 # 开头。
    """
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


def check_command_warnings(command: str) -> List[str]:
    """
    只做提示，不自动修正、不阻止运行。
    """
    warnings: List[str] = []

    if "train.py" not in command:
        warnings.append("命令中没有 train.py，可能不是训练命令。")

    if "Tee-Object" in command or "2>&1" in command:
        warnings.append("命令中仍包含 PowerShell 重定向或 Tee-Object；本脚本不会自动删除，可能导致运行失败。")

    if "--data_path" not in command:
        warnings.append("缺少 --data_path。")
    if "--data_path_right" not in command:
        warnings.append("缺少 --data_path_right。")
    if "--save_dir" not in command:
        warnings.append("缺少 --save_dir。")
    if "--shared_dataset" not in command:
        warnings.append("缺少 --shared_dataset。")
    if "--split_seed" not in command:
        warnings.append("缺少 --split_seed。")
    if "--split_fold" not in command:
        warnings.append("缺少 --split_fold。")

    return warnings


def run_one_command(command: str, log_path: Path, cwd: Path) -> int:
    """
    执行一条命令，将 stdout/stderr 同时输出到控制台和 log。
    返回进程 return code。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # 让 Python 子进程尽量实时输出。
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

        # shell=True 是为了直接运行 run.txt 中的整行 Windows 命令。
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


# =========================
# 主程序
# =========================

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

            # 确保异常也写入对应 log。
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

        # 每条跑完都更新一次 summary，防止中途断电丢记录。
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

    # 有失败也返回 0，避免 PyCharm 把整个批处理标红；具体失败看 summary。
    return 0


if __name__ == "__main__":
    sys.exit(main())
