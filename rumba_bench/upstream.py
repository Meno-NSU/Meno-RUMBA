"""Run the official pure functions without loading judge weights. / Функции авторов без весов судьи."""
import ast
import hashlib
import re
import runpy
from .data import ROOT, read_json


def load_official():
    folder = ROOT / "vendor/rumba"
    manifest = read_json(folder / "manifest.json")
    for name, info in manifest["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Upstream source changed: {name}")
    namespace = runpy.run_path(str(folder / "prompts.py"))
    namespace["re"] = re
    source = (folder / "rumba_ru.py").read_text()
    names = {"judge_template", "process_judge_response", "extract_judge_feedback"}
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("Official metric API changed")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(folder / "rumba_ru.py"), "exec"), namespace)
    return namespace


def f1_score(gold, prediction):
    """LightEval 0.13.0 F1_score defaults: case-sensitive whitespace SETS, no normalization.

    Точная семантика LightEval 0.13.0: множества слов, регистр и пунктуация сохраняются.
    """
    gold_words, predicted_words = set(gold.split()), set(prediction.split())
    if not gold_words or not predicted_words:
        return 0.0
    return 2 * len(gold_words & predicted_words) / (len(gold_words) + len(predicted_words))
