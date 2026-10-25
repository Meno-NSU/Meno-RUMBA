import importlib.util
import os
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_meno_pair", Path(__file__).resolve().parents[1] / "scripts/run_meno_pair.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


def test_missing_gpu_gres_requires_verified_exclusive_node():
    env = {"SLURM_JOB_ID": "42"}
    assert pair.allocated_devices(env, "NumNodes=1 Shared=0", 8) == [str(i) for i in range(8)]
    assert pair.allocated_devices(env, "NumNodes=1 OverSubscribe=NO", 8) == [str(i) for i in range(8)]
    for description in ("NumNodes=1 Shared=1", "NumNodes=2 Shared=0", "NumNodes=1"):
        with pytest.raises(ValueError):pair.allocated_devices(env, description, 8)
    with pytest.raises(ValueError):pair.allocated_devices({}, "NumNodes=1 Shared=0", 8)


def test_scheduler_gpu_ids_are_not_replaced_with_host_numbers():
    ids = ["GPU-" + str(i) for i in range(8)]
    assert pair.allocated_devices({"SLURM_JOB_ID": "42", "CUDA_VISIBLE_DEVICES": ",".join(ids)}, "", 8) == ids
    with pytest.raises(ValueError):pair.allocated_devices({"SLURM_JOB_ID": "42", "CUDA_VISIBLE_DEVICES": "0,1"}, "", 8)


def test_pooling_configs_without_weights_are_not_ready(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError):pair.validate_pooling_weights(tmp_path)
    (tmp_path / "pytorch_model.bin").write_bytes(b"x" * 2048)
    assert pair.validate_pooling_weights(tmp_path) == ["pytorch_model.bin"]


def test_partial_pooling_shards_are_not_ready(tmp_path):
    import json
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}))
    (tmp_path / "part1.safetensors").write_bytes(b"x" * 2048)
    with pytest.raises(ValueError):pair.validate_pooling_weights(tmp_path)
    (tmp_path / "part2.safetensors").write_bytes(b"x" * 2048)
    assert pair.validate_pooling_weights(tmp_path) == ["part1.safetensors", "part2.safetensors"]
