"""Regression checks for stale, zombie and reused PID records."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

spec = importlib.util.spec_from_file_location("server_process", Path(__file__).resolve().parents[1] / "scripts/server_process.py")
process = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process)


def test_switch_targets_only_recorded_benchmark_generators(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(process, "stop", lambda path, port, model: calls.append((path.name, port, model)))
    process.stop_benchmark(tmp_path)
    assert len(calls) == 8
    assert {name for _, _, name in calls} == {"meno-lite-0.1", "meno-lite-0.2", "Qwen3.5-9B", "Qwen2.5-7B-Instruct"}
    assert {port for _, port, _ in calls} == {9022, 9025}
    assert not any("reranker" in path or "embedder" in path for path, _, _ in calls)


@pytest.mark.parametrize("contents", ["0\n", "1\n", "not-a-pid\n", ""])
def test_invalid_pid_never_receives_a_signal(tmp_path, monkeypatch, contents):
    file = tmp_path / "server.pid"
    file.write_text(contents)
    monkeypatch.setattr(process.os, "kill", lambda *args: pytest.fail("Invalid PID must not receive a signal"))
    process.stop(file, 9022, "model", timeout=0)
    assert not file.exists()


def test_zombie_is_already_stopped_even_while_kill_zero_succeeds(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    file = tmp_path / "server.pid"
    file.write_text(str(child.pid) + "\n")
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            info = process.process_info(child.pid)
            if info and info["state"].startswith("Z"):
                break
            time.sleep(0.02)
        else:
            pytest.fail("Could not observe child zombie without reaping it")
        os.kill(child.pid, 0)
        process.stop(file, 9022, "model", timeout=0)
        assert not file.exists()
    finally:
        child.wait(timeout=5)


def test_reused_pid_does_not_kill_unrelated_process(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    file = tmp_path / "server.pid"
    file.write_text(str(child.pid) + "\n")
    try:
        process.stop(file, 9022, "model", timeout=0)
        assert child.poll() is None
        assert not file.exists()
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_recorded_service_can_be_stopped_after_process_renaming(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    file = tmp_path / "server.pid"
    try:
        process.record(file, child.pid, 9022, "model")
        process.stop(file, 9022, "model", timeout=3)
        assert not file.exists() and not process.identity_path(file).exists()
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_reused_birth_identity_is_not_signalled(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    file = tmp_path / "server.pid"
    try:
        process.record(file, child.pid, 9022, "model")
        metadata = process.identity_path(file)
        value = json.loads(metadata.read_text())
        value["start"] = "old-process-birth"
        metadata.write_text(json.dumps(value))
        process.stop(file, 9022, "model", timeout=0)
        assert child.poll() is None and not file.exists()
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_occupied_port_is_reported_instead_of_killing_its_owner():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with pytest.raises(RuntimeError, match="occupied by an untracked listener"):
            process.require_free_port(listener.getsockname()[1])
