"""Closed-user checkpoints: append-only objects and an atomic commit manifest.

Live SQLite/Qdrant databases are never copied. Only graphs explicitly closed by
the harness and their answers are committed; expensive raw LLM calls are cached
separately as immutable JSON objects, including calls for an unfinished user.
"""
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from .data import read_json, write_json

READY = ".checkpoint-ready.json"


def mark_user_closed(folder, user, suite, questions):
    """Call strictly after graph.index.close(); refuse technically bad output."""
    graph = Path(folder) / "graphs" / user
    report = read_json(graph / "build_report.json")
    if not report["consistent"] or not report["all_source_chunks_stored"] or report["llm_errors"]:
        raise ValueError(f"Cannot checkpoint a graph with failed QC: {graph}")
    paths = []
    for row in questions:
        path = Path(folder) / "answers" / suite / (row["id"].replace("|", "_") + ".json")
        answer = read_json(path)
        if (answer["id"] != row["id"] or answer["user_id"] != user or answer.get("truncated")
                or answer.get("finish_reason") != "stop" or not answer.get("prediction", "").strip()):
            raise ValueError(f"Cannot checkpoint an invalid answer: {path}")
        paths.append(path.name)
    write_json(graph / READY, {"user": user, "suite": suite, "answers": paths,
                              "closed_at": time.time(), "graph_closed": True})


def _atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial-" + uuid.uuid4().hex)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


class ClosedUserCheckpoints:
    def __init__(self, durable, fingerprint):
        self.durable = Path(durable)
        self.fingerprint = fingerprint
        self.durable.mkdir(parents=True, exist_ok=True)
        self.latest = self.durable / "latest.json"
        self.state = read_json(self.latest) if self.latest.exists() else {
            "version": 1, "fingerprint": fingerprint, "users": {}, "cache_files": 0}
        if self.state["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint fingerprint differs; use a separate run directory")
        self._cached = {}

    def _copy_cache(self, work):
        count = 0
        for kind in ("generation", "tokenization"):
            source = Path(work) / "runs/cache/ragu" / kind
            target = self.durable / "cache" / kind
            target.mkdir(parents=True, exist_ok=True)
            if kind not in self._cached:
                self._cached[kind] = {p.name for p in target.glob("*.json")}
            if source.exists():
                for path in source.glob("*.json"):
                    if path.name not in self._cached[kind]:
                        # Cache writes are atomic in the harness. This parse
                        # also refuses a truncated/corrupt source JSON object.
                        read_json(path)
                        _atomic_copy(path, target / path.name)
                        self._cached[kind].add(path.name)
            count += len(self._cached[kind])
        return count

    def commit(self, work, run_relative, labels, suite, *, final=False):
        work, run_relative = Path(work), Path(run_relative)
        if run_relative.is_absolute() or ".." in run_relative.parts:
            raise ValueError("Checkpoint run path must be relative to the working copy")
        run = work / run_relative
        cached = self._copy_cache(work)
        users = {label: dict(self.state.get("users", {}).get(label, {})) for label in labels}
        added = 0
        for label in labels:
            source = run / label
            for marker in sorted((source / "graphs").glob("*/" + READY)):
                user = marker.parent.name
                if user in users[label]:
                    continue
                ready = read_json(marker)
                if (not ready.get("graph_closed") or ready.get("suite") != suite
                        or ready.get("user") != user):
                    raise ValueError(f"Invalid closed-graph marker: {marker}")
                destination = self.durable / "objects" / label / user
                if not destination.exists():
                    temporary = destination.with_name(user + ".partial-" + uuid.uuid4().hex)
                    temporary.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copytree(marker.parent, temporary / "graph")
                        for name in ready["answers"]:
                            if Path(name).name != name:
                                raise ValueError("Invalid answer filename in checkpoint marker")
                            answer = source / "answers" / suite / name
                            row = read_json(answer)
                            if row["user_id"] != user or row.get("truncated"):
                                raise ValueError(f"Answer belongs to another user: {answer}")
                            _atomic_copy(answer, temporary / "answers" / name)
                            retrieval = source / "retrieval" / suite / name
                            if retrieval.exists():
                                _atomic_copy(retrieval, temporary / "retrieval" / name)
                        write_json(temporary / "object.json", ready)
                        temporary.replace(destination)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                # A complete object can survive a failure just before the
                # manifest commit. Validate it before adopting it on retry.
                saved = read_json(destination / "object.json")
                if saved["user"] != user or saved["suite"] != suite or saved["answers"] != ready["answers"]:
                    raise ValueError("Conflicting immutable user checkpoint")
                users[label][user] = {"object": str(destination.relative_to(self.durable)),
                                      "answers": len(saved["answers"])}
                added += 1
        # Run identity is tiny and atomically written. No live database here.
        for path in [run / "manifest.json", *[run / label / "endpoint.json" for label in labels]]:
            if path.exists():
                _atomic_copy(path, self.durable / "identity" / path.relative_to(run))
        state = {"version": 1, "fingerprint": self.fingerprint, "time": time.time(),
                 "run_relative": str(run_relative), "suite": suite, "users": users,
                 "cache_files": cached, "final": final}
        # Publish only after every object is fully copied and renamed. Keep the
        # previous committed manifest; objects/cache themselves are append-only.
        if self.latest.exists():
            _atomic_copy(self.latest, self.durable / "previous.json")
        write_json(self.latest, state)
        self.state = state
        return {"new_users": added, "cache_files": cached,
                "users": {label: len(value) for label, value in users.items()},
                "answers": {label: sum(v["answers"] for v in value.values()) for label, value in users.items()},
                "final": final}

    def restore(self, work, run_relative):
        work, run_relative = Path(work), Path(run_relative)
        if not self.latest.exists():
            return {"users": 0, "answers": 0}
        if str(run_relative) != self.state["run_relative"]:
            raise ValueError("Checkpoint RAGU manifest/run identity changed")
        run = work / run_relative
        identity = self.durable / "identity"
        if identity.exists():
            shutil.copytree(identity, run, dirs_exist_ok=True)
        count = answers = 0
        for label, users in self.state["users"].items():
            for user, entry in users.items():
                relative = Path(entry["object"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid checkpoint object path")
                source = self.durable / relative
                saved = read_json(source / "object.json")
                if saved["user"] != user or len(saved["answers"]) != entry["answers"]:
                    raise ValueError("Incomplete checkpoint object")
                target = run / label
                shutil.copytree(source / "graph", target / "graphs" / user, dirs_exist_ok=True)
                # The resumed worker will reopen this graph. Its old committed
                # object remains immutable; mark the live copy only on close.
                (target / "graphs" / user / READY).unlink(missing_ok=True)
                for name in saved["answers"]:
                    _atomic_copy(source / "answers" / name, target / "answers" / self.state["suite"] / name)
                    retrieval = source / "retrieval" / name
                    if retrieval.exists():
                        _atomic_copy(retrieval, target / "retrieval" / self.state["suite"] / name)
                count += 1
                answers += entry["answers"]
        cache = self.durable / "cache"
        if cache.exists():
            shutil.copytree(cache, work / "runs/cache/ragu", dirs_exist_ok=True)
        return {"users": count, "answers": answers}

    def export_answers(self, destination):
        """Expose judge-ready answers without duplicating graph/cache gigabytes."""
        destination = Path(destination) / self.state["run_relative"]
        if (self.durable / "identity").exists():
            shutil.copytree(self.durable / "identity", destination, dirs_exist_ok=True)
        for label, users in self.state["users"].items():
            for user, entry in users.items():
                source = self.durable / entry["object"]
                saved = read_json(source / "object.json")
                for name in saved["answers"]:
                    target = destination / label / "answers" / self.state["suite"] / name
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.link(source / "answers" / name, target)
        return destination
