"""Dataset preparation and fixed pilot. / Подготовка данных и фиксированный пилот."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def category(row):
    parts = [part.strip() for part in row["category"].split(",")]
    if len(parts) != 3 or parts[1] not in ("single", "multi") or parts[2] not in ("temporal", "atemporal"):
        raise ValueError(f"Unknown RUMBA category: {row['category']}")
    return parts[0], parts[1], parts[2]


def timestamp(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def pilot_ids(questions, count=50, seed=42):
    """Use few complete user histories, then balance four cells. / Мало пользователей, четыре группы."""
    rng = random.Random(seed)
    users = sorted({q["user_id"] for q in questions})
    rng.shuffle(users)
    pool = []
    picked_users = set()
    for user in users:
        picked_users.add(user)
        pool.extend(q for q in questions if q["user_id"] == user)
        cells = {category(q)[1:] for q in pool}
        if len(pool) >= count and len(cells) == 4:
            break
    if len(pool) < count:
        raise ValueError("Not enough questions for pilot")
    groups = defaultdict(list)
    for q in sorted(pool, key=lambda q: q["id"]):
        groups[category(q)[1:]].append(q)
    for group in groups.values():
        rng.shuffle(group)
    chosen = []
    while len(chosen) < count:
        for key in sorted(groups):
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
    return [q["id"] for q in chosen]


def prepare():
    import os
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    from datasets import load_dataset
    manifest = read_json(ROOT / "vendor/rumba/manifest.json")
    target = ROOT / "data"
    if (target / "manifest.json").exists():
        print(canonical(read_json(target / "manifest.json")))
        return
    target.mkdir(exist_ok=True)
    questions, conversations = [], {}
    ds = load_dataset("ai-forever/RUMBA", "ru", split="test", streaming=True,
                      revision=manifest["dataset_revision"])
    for row in ds:
        row = json.loads(canonical(row))
        history = row.pop("conversation")
        user = row["user_id"]
        if user in conversations and digest(conversations[user]) != digest(history):
            raise ValueError(f"Different histories for {user}; implement per-question snapshots first")
        conversations[user] = history
        category(row)
        questions.append(row)
        if len(questions) % 100 == 0:
            print(f"Prepared {len(questions)} questions / вопросов", flush=True)
    if len(questions) != 1543 or len({q['id'] for q in questions}) != 1543:
        raise ValueError(f"Expected 1543 unique questions, got {len(questions)}")
    qpath = target / "questions.jsonl"
    qpath.write_text("".join(canonical(q) + "\n" for q in questions))
    for user, history in conversations.items():
        write_json(target / "conversations" / f"{user}.json", history)
    ids = pilot_ids(questions)
    write_json(target / "pilot50.json", ids)
    selected = [q for q in questions if q["id"] in ids]
    stats = {
        "dataset": "ai-forever/RUMBA", "config": "ru", "split": "test",
        "revision": manifest["dataset_revision"], "questions_sha256": digest(questions),
        "histories_sha256": digest(conversations), "questions": len(questions),
        "users": len(conversations), "sessions": sum(map(len, conversations.values())),
        "cells": dict(Counter(f"{category(q)[2]} × {category(q)[1]}" for q in questions)),
        "pilot": {"seed": 42, "ids": ids, "users": sorted({q['user_id'] for q in selected}),
                  "cells": dict(Counter(f"{category(q)[2]} × {category(q)[1]}" for q in selected)),
                  "selection": "few-user pool; balanced temporal × scope; not a representative global estimate"},
    }
    write_json(target / "manifest.json", stats)
    print(canonical(stats))
