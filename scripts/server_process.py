#!/usr/bin/env python3
"""PID lifecycle checks for the server launchers; standard library only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time


def report(message):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def process_info(pid):
    if pid <= 1:
        return None
    if Path("/proc").is_dir():
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
            return {"pid": pid, "state": fields[0], "start": fields[19], "argv": [v for v in argv if v]}
        except (FileNotFoundError, ProcessLookupError):
            return None
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat=", "-o", "lstart=", "-o", "args="],
                            capture_output=True, text=True)
    if result.returncode:
        if "not permitted" in result.stderr.lower():
            raise RuntimeError("Cannot inspect process state: " + result.stderr.strip())
        return None
    parts = result.stdout.strip().split(None, 6)
    if len(parts) < 6:
        return None
    return {"pid": pid, "state": parts[0], "start": " ".join(parts[1:6]),
            "argv": shlex.split(parts[6]) if len(parts) > 6 else []}


def alive(info):
    return bool(info and info["state"][0] not in "ZXx")


def identity_path(path):
    return Path(str(path) + ".identity.json")


def discard(path):
    Path(path).unlink(missing_ok=True)
    identity_path(path).unlink(missing_ok=True)


def record(path, pid, port, model):
    info = process_info(pid)
    if not alive(info):
        raise RuntimeError(f"New process {pid} already exited; inspect its vLLM log")
    Path(path).write_text(str(pid) + "\n")
    identity_path(path).write_text(json.dumps({"pid": pid, "start": info["start"], "port": port, "model": model}) + "\n")
    report(f"Recorded PID {pid}, state={info['state']}, start={info['start']}, model={model}, port={port}")


def expected_process(info, path, port, model):
    sidecar = identity_path(path)
    if sidecar.exists():
        try:
            saved = json.loads(sidecar.read_text())
            return all(saved.get(k) == v for k, v in {
                "pid": info["pid"], "start": info["start"], "port": port, "model": model}.items())
        except (ValueError, OSError):
            return False
    # Compatibility with old, single-line PID files: require the actual server
    # command, expected port and served model. Bare renamed processes are not
    # enough evidence to send a signal without a recorded birth identity.
    args = info["argv"]
    command = " ".join(args).lower()
    def option(name, value):
        return f"{name}={value}" in args or any(a == name and b == value for a, b in zip(args, args[1:]))
    return ("vllm" in command and ("serve" in args or "vllm.entrypoints.openai.api_server" in command)
            and option("--port", str(port)) and option("--served-model-name", model))


def stop(path, port, model, timeout=30):
    path = Path(path)
    if not path.exists():
        return
    try:
        pid = int(path.read_text().strip())
    except ValueError:
        pid = 0
    info = process_info(pid)
    if not alive(info):
        report(f"Removing stale PID record {path} (PID {pid}, state {info['state'] if info else 'absent'})")
        discard(path)
        return
    if not expected_process(info, path, port, model):
        report(f"PID {pid} no longer identifies {model} on port {port}; leaving the process untouched")
        discard(path)
        return
    original_start = info["start"]
    current = process_info(pid)
    if not alive(current) or current["start"] != original_start:
        discard(path)
        return
    try:
        report(f"Sending SIGTERM to PID {pid}, state={current['state']}, model={model}, port={port}")
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        discard(path)
        return
    deadline = time.monotonic() + timeout
    while True:
        current = process_info(pid)
        if not alive(current) or current["start"] != original_start:
            report(f"PID {pid} stopped or was replaced; clearing its old PID record")
            discard(path)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Process {pid} is still running (state {current['state']}); inspect it before restarting")
        time.sleep(0.2)


def require_free_port(port):
    with socket.socket() as client:
        client.settimeout(1)
        if client.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is still occupied by an untracked listener; inspect it before restarting")
    report(f"Port {port} is free")


def stop_benchmark(launch_dir, native_port=9022, long_port=9025):
    """Switch only known recorded generators; pooling services stay running."""
    for tag, model in (("meno2", "meno-lite-0.2"), ("meno1", "meno-lite-0.1"),
                       ("qwen35", "Qwen3.5-9B"), ("qwen25", "Qwen2.5-7B-Instruct")):
        stop(Path(launch_dir) / f"vllm_process_{tag}.txt", native_port, model)
        stop(Path(launch_dir) / f"vllm_process_{tag}_yarn.txt", long_port, model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stop", "record", "alive", "free", "stop-benchmark"))
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--model")
    parser.add_argument("--launch-dir", type=Path)
    parser.add_argument("--long-port", type=int, default=9025)
    args = parser.parse_args()
    try:
        if args.action == "stop-benchmark":
            if args.launch_dir is None:
                raise ValueError("stop-benchmark requires --launch-dir")
            stop_benchmark(args.launch_dir, args.port or 9022, args.long_port)
        elif args.action == "stop":
            stop(args.pid_file, args.port, args.model)
        elif args.action == "record":
            record(args.pid_file, args.pid, args.port, args.model)
        elif args.action == "alive":
            raise SystemExit(0 if alive(process_info(args.pid)) else 1)
        else:
            require_free_port(args.port)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2 if args.action == "alive" else 1, f"{error}\n")


if __name__ == "__main__":
    main()
