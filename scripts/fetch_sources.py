"""Pin the public upstream sources. / Зафиксировать открытые исходники авторов."""
import hashlib
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "vendor" / "rumba"


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "meno-rumba-benchmark"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / "manifest.json"
    if manifest_path.exists():
        print(manifest_path.read_text())
        return
    commit = json.loads(get("https://api.github.com/repos/ai-forever/RUMBA/commits/main"))["sha"]
    paths = {
        "rumba_ru.py": "evaluation/lighteval_ext/custom_tasks/rumba_ru.py",
        "prompts.py": "evaluation/prompts.py",
        "LICENSE.txt": "LICENSE.txt",
    }
    manifest = {"repository": "https://github.com/ai-forever/RUMBA", "commit": commit, "files": {}}
    for name, remote in paths.items():
        raw = get(f"https://raw.githubusercontent.com/ai-forever/RUMBA/{commit}/{remote}")
        (DEST / name).write_bytes(raw)
        manifest["files"][name] = {"path": remote, "sha256": hashlib.sha256(raw).hexdigest()}
    hf = json.loads(get("https://huggingface.co/api/datasets/ai-forever/RUMBA"))
    manifest["dataset_revision"] = hf["sha"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
