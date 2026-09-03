#!/usr/bin/env python3
"""Claude Code PreToolUse hook: refuse tool calls that would show or transmit
row-level iCTG patient data.

Reads the hook JSON from stdin and prints a permission decision. Deterministic,
stdlib only (must run on the system python3). Data roots come from a built-in
list plus any CTG_* path environment variables and CTG_GUARD_EXTRA_ROOTS
(colon separated).

Blocked when a command/path touches a data root:
  * printing or paging rows (cat, head, tail, less, sed, awk, grep, ...)
  * query/DataFrame idioms that return rows (SELECT *, LIMIT, .head(), ...)
  * sending or copying the data outside the data roots (curl, scp, rsync, cp ...)
  * Read/Grep tool on data files (parquet, csv, xlsx, json, zip, duckdb, ...)
Everything else is allowed, with a reminder injected when a data root is named.
"""

from __future__ import annotations

import json
import os
import re
import sys

BUILTIN_ROOTS = ["/srv/data/input/iCTG", "/srv/data/iCTG"]
DATA_EXT = r"(parquet|csv|tsv|xlsx|xls|json|jsonl|zip|gz|duckdb|db|pkl|pickle|npy|npz|pt|feather|arrow)"

ROW_TOOLS = (
    r"\b(cat|head|tail|less|more|most|nl|tac|xxd|hexdump|od|strings|column|bat|batcat|"
    r"vim?|nvim|nano|emacs|sed|awk|gawk|mawk|perl|grep|egrep|fgrep|rg|ag|ack|"
    r"csvlook|csvcut|csvgrep|csvjson|xsv|qsv|mlr|miller|jq|yq|"
    r"zcat|gzcat|gunzip|bzcat|xzcat|unzip|7z|tar|"
    r"parquet-tools|parquet-cli|pqrs|pqviewer|duckdb|sqlite3|in2csv|xlsx2csv|libreoffice|soffice)\b"
)
ROW_IDIOMS = [
    r"select\s+\*",
    r"select\s+(?!count\s*\().*?\bfrom\b(?![^;]*\bgroup\s+by\b)(?![^;]*\bcount\s*\()",
    r"\blimit\s+\d+",
    r"\bsample\s+\d+",
    r"\.head\s*\(",
    r"\.tail\s*\(",
    r"\.sample\s*\(",
    r"\.to_string\s*\(",
    r"\.to_markdown\s*\(",
    r"\.to_pylist\s*\(",
    r"\.to_dict\s*\(",
    r"\.iloc\b",
    r"\.loc\[",
    r"\.show\s*\(",
    r"\.describe\s*\(",
    r"\.unique\s*\(",
    r"\.drop_duplicates\s*\(",
    r"\.value_counts\s*\(",
    r"\.slice\s*\(",
    r"\.take\s*\(",
    r"read_(csv|parquet|excel|json|table)\s*\([^)]*\)\s*\)?\s*$",  # bare read at end of -c
    r"\bprint\s*\(\s*(df|table|tbl|rows|row|batch|record|records|pf|pq_file)\b",
    r"iter_batches|iterrows|itertuples|to_batches",
]
XFER_TOOLS = (
    r"\b(curl|wget|scp|sftp|ssh|rsync|rclone|nc|ncat|netcat|socat|ftp|lftp|telnet|"
    r"aws|gsutil|gcloud|az|b2|s3cmd|mc|"
    r"mail|mailx|sendmail|mutt|msmtp|"
    r"gh|hub|git\s+add|git\s+commit|git\s+push|git\s+lfs|"
    r"cp|mv|ln|install|dd|tee|tar|zip|7z|gzip|bzip2|xz|zstd|base64|split)\b"
)
OUTSIDE_HINTS = r"(~|\$HOME|/home/|/tmp/|/root/|/mnt/|/media/|/var/|/opt/|\./|\.\./|\$PWD|\$\(pwd\))"


def data_roots() -> list[str]:
    roots = list(BUILTIN_ROOTS)
    for key, val in os.environ.items():
        if key.startswith("CTG_") and val.startswith("/"):
            roots.append(val)
    roots += [p for p in os.environ.get("CTG_GUARD_EXTRA_ROOTS", "").split(":") if p]
    return sorted({r.rstrip("/") for r in roots}, key=len, reverse=True)


def deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "PATIENT DATA GUARD: " + reason
                        + " Row-level iCTG data must never be displayed or sent anywhere. "
                        "Use schema/metadata, COUNT(*)/aggregates, or equality joins that "
                        "report how many rows match (see CLAUDE.md)."
                    ),
                }
            }
        )
    )
    sys.exit(0)


def allow(context: str | None = None) -> None:
    if context:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "additionalContext": context,
                    }
                }
            )
        )
    sys.exit(0)


def under_root(path: str, roots: list[str]) -> bool:
    path = os.path.abspath(os.path.expanduser(path)) if path else ""
    return any(path == r or path.startswith(r + "/") for r in roots)


def mentions_root(text: str, roots: list[str]) -> bool:
    return any(r in text for r in roots)


def check_bash(command: str, roots: list[str]) -> None:
    cmd = command
    low = cmd.lower()
    touches = mentions_root(cmd, roots)
    # A parquet/duckdb file anywhere is treated as potential patient data.
    touches_binary = re.search(r"\S+\.(parquet|duckdb)\b", low) is not None
    if not (touches or touches_binary):
        allow()

    if re.search(ROW_TOOLS, cmd):
        # DuckDB/sqlite CLIs are fine only for aggregates; other tools never.
        if re.search(r"\b(duckdb|sqlite3)\b", cmd):
            for idiom in ROW_IDIOMS:
                if re.search(idiom, low, flags=re.S):
                    deny("query returns rows, not aggregates.")
        else:
            deny("command prints or pages file content.")

    for idiom in ROW_IDIOMS:
        if re.search(idiom, low, flags=re.S):
            deny("code idiom returns/prints rows, not aggregates.")

    if touches and re.search(XFER_TOOLS, cmd):
        m = re.search(XFER_TOOLS, cmd)
        tool = m.group(1) if m else "tool"
        copy_like = tool in {"cp", "mv", "ln", "install", "dd", "tee", "tar", "zip", "7z",
                             "gzip", "bzip2", "xz", "zstd", "base64", "split"}
        if not copy_like or re.search(OUTSIDE_HINTS, cmd) or not all(
            under_root(p, roots) for p in re.findall(r"(/\S+)", cmd)
        ):
            deny(f"'{tool}' would copy or transmit patient data outside the data roots.")

    allow(
        "Reminder: this command touches iCTG patient data. Output only counts and "
        "aggregates; never rows, IDs or free text."
    )


def check_file_tool(tool: str, inp: dict, roots: list[str]) -> None:
    path = inp.get("file_path") or inp.get("path") or ""
    if path and under_root(path, roots):
        if tool == "Grep":
            deny("Grep over the data directory would print matching rows.")
        if re.search(r"\.(" + DATA_EXT + r")$", path, flags=re.I):
            deny(f"Read of data file '{os.path.basename(path)}'.")
        allow("Reminder: this file lives under the iCTG data root; do not quote patient content.")
    allow()


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        allow()
    tool = payload.get("tool_name", "")
    inp = payload.get("tool_input") or {}
    roots = data_roots()
    if tool == "Bash":
        check_bash(str(inp.get("command", "")), roots)
    elif tool in ("Read", "Grep", "Edit", "Write", "NotebookEdit"):
        check_file_tool(tool, inp, roots)
    allow()


if __name__ == "__main__":
    main()
