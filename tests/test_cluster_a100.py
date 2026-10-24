"""Cluster GPU ownership and insufficient-resource checks without CUDA or Slurm."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cluster_a100.sh"


@pytest.mark.parametrize("slurm,mask,layout,expected", [
    (False, "0", "single80", "1"),
    (True, "0", "single80", "0"),
    (True, "GPU-allocated-uuid", "single80", "GPU-allocated-uuid"),
    (True, "2,4", "dual40", "2,4"),
])
def test_gpu_mask_is_preserved_under_slurm_and_memory_checked_before_launch(tmp_path, slurm, mask, layout, expected):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(SCRIPT, scripts / SCRIPT.name)
    for name in (".venv-ragu", ".venv-vllm"):
        bin_dir = tmp_path / name / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python").symlink_to(sys.executable)
    server = tmp_path / ".venv-vllm/bin/vllm"
    server.write_text('#!/bin/sh\ntouch "$UNEXPECTED_SERVER_START"\nexit 99\n')
    server.chmod(0o755)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text("{}")
    mocks = tmp_path / "mocks"
    mocks.mkdir()
    (mocks / "torch.py").write_text('''
import os
from pathlib import Path
from types import SimpleNamespace
def count():
    Path(os.environ["MASK_TRACE"]).write_text(os.environ["CUDA_VISIBLE_DEVICES"])
    return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
cuda = SimpleNamespace(device_count=count, get_device_properties=lambda i:
    SimpleNamespace(name="NVIDIA A100", total_memory=1024**3))
''')
    trace = tmp_path / "mask.txt"
    unexpected = tmp_path / "unexpected-start"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": mask, "A100_LAYOUT": layout,
           "PYTHONPATH": str(mocks), "MASK_TRACE": str(trace),
           "UNEXPECTED_SERVER_START": str(unexpected)}
    env.pop("SLURM_JOB_ID", None)
    if slurm:
        env["SLURM_JOB_ID"] = "test-job"
    result = subprocess.run(["bash", str(scripts / SCRIPT.name), "inference", "meno-lite-0.2"],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "requires full A100 devices" in result.stdout + result.stderr
    assert trace.read_text() == expected
    assert not unexpected.exists()


@pytest.mark.parametrize("slurm,mask,layout,message", [
    (True, None, "single80", "Slurm must expose the allocated GPUs"),
    (False, "0,1", "dual40", "dual40 requires a two-GPU Slurm allocation"),
    (True, "MIG-allocated-instance", "single80", "not MIG instances"),
])
def test_unsupported_gpu_allocations_fail_early(tmp_path, slurm, mask, layout, message):
    env = {**os.environ, "A100_LAYOUT": layout}
    env.pop("SLURM_JOB_ID", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if slurm:
        env["SLURM_JOB_ID"] = "test-job"
    if mask is not None:
        env["CUDA_VISIBLE_DEVICES"] = mask
    result = subprocess.run(["bash", str(SCRIPT), "inference", "meno-lite-0.2"],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert message in result.stdout + result.stderr
