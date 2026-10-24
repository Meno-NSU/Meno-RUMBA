"""Replay failed RAGU messages under the current, explicitly recorded decoder.

This is a diagnostic regression check. It never substitutes answers in a run.
"""
import argparse
import asyncio
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragu.common.prompts.default_models import EntitiesExtractionModel, RelationsExtractionModel
from rumba_bench.data import canonical, digest, read_json, write_json
from rumba_bench.ragu_harness import ProtocolLLM


async def run(args):
    config = read_json(args.config)
    llm = ProtocolLLM(config["models"][args.model], config, ROOT / "runs/cache/ragu/generation")
    models = {model.__name__: model for model in (EntitiesExtractionModel, RelationsExtractionModel)}
    rows = []
    try:
        for path in args.responses:
            baseline = read_json(path)
            schema = baseline["request"]["extra_body"]["structured_outputs"]["json"]
            model = models[schema["title"]]
            started = time.monotonic()
            row = {"baseline_cache_file": str(path), "baseline_finish_reason": baseline["finish_reason"],
                   "baseline_completion_tokens": baseline["usage"].get("completion_tokens"),
                   "conversation_hash": digest(baseline["request"]["messages"])}
            try:
                value = await llm.chat_completion(baseline["request"]["messages"], output_schema=model)
                raw = llm.calls[-1]
                items = getattr(value, "entities" if model is EntitiesExtractionModel else "relations")
                row.update({"status": "ok", "items": len(items), "unique_items": len({item.model_dump_json() for item in items}),
                            "finish_reason": raw["finish_reason"], "usage": raw["usage"], "generation_seconds": raw["seconds"],
                            "raw_response_hash": digest(raw)})
                if not items:
                    row["status"] = "empty"
            except Exception as error:
                row.update({"status": "error", "error": str(error)})
            row["elapsed_seconds"] = time.monotonic() - started
            rows.append(row)
            print(canonical(row), flush=True)
            write_json(args.output, {"model": config["models"][args.model], "protocol": config["protocol"], "rows": rows})
    finally:
        llm.close()
    if not rows or any(row["status"] != "ok" for row in rows):
        raise RuntimeError(f"Decoder regression failed; see {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ragu.json")
    parser.add_argument("--model", default="meno-lite-0.2")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/ragu-decoder-regression.json")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
