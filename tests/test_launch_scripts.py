"""Exercise the real shell launch flow with CPU-only process and HTTP stubs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path, contents):
    path.write_text(contents)
    path.chmod(0o755)


@pytest.mark.parametrize("script", [
    "launch_split_context.sh", "restart_meno2_split_context.sh",
    "launch_meno2.sh", "launch_reranker_8192.sh",
])
@pytest.mark.parametrize("fail_start", [False, True])
def test_all_server_launchers_force_gpu_one_and_log_failures(tmp_path, script, fail_start):
    launch = tmp_path / "launch"
    launch.mkdir()
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    activate = launch / "vllm_0_29_0/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    trace = tmp_path / "launches.jsonl"
    config_cuda = tmp_path / "config-cuda.txt"
    executable(fakebin / "python", """#!/bin/sh
case "$1" in
  */server_process.py) exec "$FAKE_REAL_PYTHON" "$@" ;;
esac
printf '%s' "$CUDA_VISIBLE_DEVICES" > "$FAKE_CONFIG_CUDA"
printf '%s\\n' '{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768,"rope_theta":1000000}}'
""")
    executable(fakebin / "sleep", "#!/bin/sh\nexit 0\n")
    executable(fakebin / "vllm", f"#!{sys.executable}\n" + """
import json, os, sys, time
if os.environ.get('FAKE_VLLM_FAIL') == '1':
    print('FAKE_VLLM_START_FAILURE', file=sys.stderr, flush=True)
    sys.exit(9)
with open(os.environ['FAKE_LAUNCH_TRACE'], 'a') as handle:
    handle.write(json.dumps({'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'), 'args': sys.argv[1:]}) + '\\n')
time.sleep(10)
""")
    executable(fakebin / "curl", f"#!{sys.executable}\n" + """
import json, os, sys
from urllib.parse import urlparse
port = str(urlparse(sys.argv[-1]).port)
try:
    with open(os.environ['FAKE_LAUNCH_TRACE']) as handle:
        for line in handle:
            try:
                args = json.loads(line)['args']
                if args[args.index('--port') + 1] == port:
                    sys.exit(0)
            except (ValueError, KeyError):
                pass
except FileNotFoundError:
    pass
sys.exit(1)
""")
    pidfiles = [tmp_path / name for name in ("native.pid", "yarn.pid", "reranker.pid")]
    env = {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
           "VLLM_BIN": str(fakebin / "vllm"), "CUDA_VISIBLE_DEVICES": "0", "GPU_ID": "0",
           "FAKE_LAUNCH_TRACE": str(trace), "FAKE_CONFIG_CUDA": str(config_cuda),
           "FAKE_REAL_PYTHON": sys.executable,
           "FAKE_VLLM_FAIL": "1" if fail_start else "0",
           "VLLM_LAUNCH_DIR": str(launch), "VLLM_ENV_DIR": str(activate.parent.parent),
           "NATIVE_PID_FILE": str(pidfiles[0]), "YARN_PID_FILE": str(pidfiles[1]),
           "RERANKER_PID_FILE": str(pidfiles[2])}
    args = ["bash", str(ROOT / "scripts" / script)]
    if script == "launch_split_context.sh":
        args.extend([str(tmp_path / "models/meno-lite-0.2"), "meno-lite-0.2"])
    try:
        completed = subprocess.run(args, cwd=launch, env=env, capture_output=True, text=True, timeout=10)
        if fail_start:
            assert completed.returncode != 0
            logs = list(launch.glob("*launcher.log"))
            assert logs
            assert all("FAKE_VLLM_START_FAILURE" in path.read_text() and "EXIT: code=1" in path.read_text()
                       and "exit code=9" in path.read_text() for path in logs)
            return
        assert completed.returncode == 0, completed.stdout + completed.stderr
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        assert all(row["cuda"] == "1" for row in rows)
        ports = [row["args"][row["args"].index("--port") + 1] for row in rows]
        if script == "launch_reranker_8192.sh":
            assert ports == ["9024"]
        else:
            assert ports == ["9022", "9025"]
            assert config_cuda.read_text() == "1"
            assert "--hf-overrides" not in rows[0]["args"]
            assert "--hf-overrides" in rows[1]["args"]
        logs = list(launch.glob("*launcher.log"))
        assert logs and all("START:" in path.read_text() and "COMMAND:" in path.read_text() for path in logs)
    finally:
        for path in pidfiles:
            if path.exists():
                try:
                    os.kill(int(path.read_text().strip()), signal.SIGTERM)
                except ProcessLookupError:
                    pass
