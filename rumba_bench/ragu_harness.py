"""Pinned upstream RAGU benchmark; all neural inference uses remote services.

The adapters below only handle transport, protocol flags and on-disk paths.
Extraction, graph construction, hybrid retrieval and reranking are RAGU's.
"""
import argparse
import asyncio
import copy
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI
from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.prompts.icl_config import ICLConfig
from ragu.common.prompts.messages import ChatMessages, UserMessage
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.index import StorageArguments
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLM
from ragu.models.openai import CachedAsyncOpenAI
from ragu.models.scorer import ScorerOpenAI
from ragu.models.sparse_embedder import BM25
from ragu.search_engine.local_search import LocalParams, LocalSearchEngine
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage
from ragu.common.logger import logger
from ragu.triplet.ontology import OntologyValidator
from ragu.triplet.two_stage_extractor import TwoStageArtifactsExtractorLLM

from .data import ROOT, canonical, digest, read_json, read_jsonl, timestamp, write_json
from .pipeline import Generation, chunks, generation_parameters
from .report import comparison, make_report
from .upstream import load_official
from .ragu_decoding import schema_grammar

RAGU_REVISION = "abd3f2958c186dc624bbd42560ad6e1e7033efaf"


def generation_schema(output_schema, protocol):
    """Optional finite decoding bounds; retain upstream field names and types."""
    if output_schema is str or protocol["extraction_decoding"] == "free":
        return None
    schema = copy.deepcopy(output_schema.model_json_schema())
    if protocol["extraction_decoding"] == "native":
        return schema
    properties = schema.get("properties", {})
    if not ({"entities", "relations"} & properties.keys()):
        return schema
    limits = protocol["structured_limits"]
    for field in ("entities", "relations"):
        if field in properties:
            properties[field]["maxItems"] = limits[field]
    if protocol["extraction_decoding"] == "bounded":
        for definition in schema.get("$defs", {}).values():
            for name, field in definition.get("properties", {}).items():
                if field.get("type") == "string":
                    field["maxLength"] = limits["description_chars" if name == "description" else "name_chars"]
                    field["minLength"] = 1
    return schema


class ProtocolLLM(LLM):
    """RAGU LLM interface with explicit vLLM flags and complete raw-response cache."""

    def __init__(self, endpoint, config, cache, *, answer=False):
        self.protocol = config["protocol"]
        # A server-side decoder change must invalidate even identical request
        # bodies: old incomplete JSON must not be reused after a redeployment.
        profile = self.protocol.get("server_structured_outputs")
        identity = {**endpoint, "server_structured_outputs": profile} if profile else endpoint
        self.generator = Generation(identity, config["runtime"], cache, self.protocol)
        self.semaphore = asyncio.Semaphore(config["runtime"]["concurrency"])
        self.max_tokens = self.protocol["answer_max_tokens" if answer else "extraction_max_tokens"]
        self.stop = self.protocol["answer_stop"] if answer else None
        self.calls, self.errors = [], []

    async def chat_completion(self, conversation, output_schema=str, **kwargs):
        # Upstream cannot silently override the common comparison protocol.
        if kwargs:
            raise ValueError(f"Unspecified generation overrides: {sorted(kwargs)}")
        schema = generation_schema(output_schema, self.protocol)
        constraint = None if schema is None else (
            {"grammar": schema_grammar(schema, self.protocol["grammar_whitespace_chars"])}
            if self.protocol["extraction_decoding"] == "grammar" else {"json": schema})
        async with self.semaphore:
            try:
                raw = await asyncio.to_thread(
                    self.generator.call, conversation, self.max_tokens,
                    stop=self.stop, seed=self.protocol["seed"], structured_outputs=constraint,
                    allow_incomplete=output_schema is str,
                )
                self.calls.append(raw)
                return raw["text"] if schema is None else output_schema.model_validate_json(raw["text"])
            except Exception as error:
                self.errors.append({"error": str(error), "conversation_hash": digest(conversation),
                                    "schema": output_schema.__name__})
                write_json(self.generator.cache / "errors" / (digest(conversation) + ".json"),
                           {"conversation": conversation, "error": str(error), "schema": schema})
                raise

    def close(self):
        self.generator.client.close()


class SeparateQdrantStorage(QdrantVectorDBStorage):
    """Give each native vector collection its own local Qdrant lock/path."""

    def __init__(self, embedding_dim, filename="data.json", storage_folder=None, **kwargs):
        filename_path = Path(storage_folder or Settings.storage_folder) / filename
        super().__init__(embedding_dim=embedding_dim, filename=filename,
                         storage_folder=storage_folder,
                         path=str(filename_path.with_suffix(".qdrant")), **kwargs)


class AuditedValidator(OntologyValidator):
    """Record native ontology decisions without changing them."""

    def __init__(self, original):
        super().__init__(original.ontology, original.policies)
        self.reports = []

    def validate_batch(self, *args, **kwargs):
        result = super().validate_batch(*args, **kwargs)
        report = result[2]
        self.reports.append({"summary": report.summary(), "breakdown": report.breakdown()})
        return result


def source_documents(user, sessions, protocol):
    """Keep roles, dates and provenance in EVERY chunk, without reading questions."""
    documents, provenance = [], {}
    for session in sorted(sessions, key=lambda s: (timestamp(s["session_datetime"]), s["session_idx"])):
        turns, offsets, offset = [], [], 0
        for message in session["conversation"]:
            text = f"{message['role']}: {message['message']}"
            offsets.append((offset, message["role"]))
            turns.append(text)
            offset += len(text) + 1
        body = "\n".join(turns)
        start = 0
        for part, text in enumerate(chunks(body, protocol["chunk_chars"], protocol["chunk_overlap_chars"])):
            role = next(role for offset, role in reversed(offsets) if offset <= start)
            source = (f"User ID: {user}\nSession ID: {session['session_idx']}\n"
                      f"Session datetime: {session['session_datetime']}\nFragment: {part}\n"
                      f"Fragment starts in role: {role}\n"
                      "user = пользователь; assistant = ассистент.\n" + text)
            documents.append(source)
            provenance[Chunk(source, len(documents) - 1, doc_id=f"doc_{len(documents) - 1}").id] = session["session_idx"]
            start += len(text) - protocol["chunk_overlap_chars"]
    return documents, provenance


def answer_instruction(question):
    """Official RU answer instructions, rendered by the actual RAGU engine."""
    prompt = load_official()["ANSWER_PROMPT_RU"]
    # Literal date values are bound here; query/context remain RAGU variables.
    bindings = ("{% set memories=[context] %}{% set question=query %}"
                "{% set query_date=" + json.dumps(question["question_date"], ensure_ascii=False) + " %}"
                "{% set query_weekday=" + json.dumps(timestamp(question["question_date"]).strftime("%A")) + " %}")
    return RAGUInstruction(messages=ChatMessages.from_messages([UserMessage(content=bindings + prompt)]))


def manifest(config):
    revision = subprocess.check_output(["git", "-C", str(ROOT / "third_party/RAGU"), "rev-parse", "HEAD"], text=True).strip()
    if revision != RAGU_REVISION:
        raise ValueError(f"RAGU revision differs: {revision}; expected {RAGU_REVISION}")
    source = ROOT / "third_party/RAGU/ragu"
    return {"harness": "ragu-two-stage-local-rerank-v1", "ragu_revision": revision,
            "ragu_source_hash": digest({str(p.relative_to(source)): digest(p.read_text()) for p in sorted(source.rglob("*.py"))}),
            "runner_hashes": {name: digest((ROOT / "rumba_bench" / name).read_text()) for name in
                              ("ragu_harness.py", "ragu_decoding.py", "pipeline.py", "data.py", "upstream.py", "report.py")},
            "protocol": config["protocol"], "generation": generation_parameters(config["protocol"]),
            "embedding": config["embedding"], "reranker": config["reranker"],
            "dataset": read_json(ROOT / "data/manifest.json"), "official": read_json(ROOT / "vendor/rumba/manifest.json"),
            "packages": {p: importlib.metadata.version(p) for p in ("graph_ragu", "openai", "qdrant-client", "fastembed", "datasets")}}


def validate_config(config):
    protocol = config["protocol"]
    if (protocol["temperature"], protocol["top_p"], protocol["seed"]) != (0, 1, 42):
        raise ValueError("This comparison requires greedy decoding and seed 42")
    generation_parameters(protocol)
    if protocol["extraction_decoding"] not in ("free", "native", "bounded", "grammar"):
        raise ValueError("extraction_decoding must be free, native, bounded or grammar")
    if type(protocol["grammar_whitespace_chars"]) is not int or not 0 <= protocol["grammar_whitespace_chars"] <= 16:
        raise ValueError("grammar_whitespace_chars must be an integer in [0, 16]")
    limits = protocol["structured_limits"]
    if set(limits) != {"entities", "relations", "name_chars", "description_chars"} or any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError("structured_limits must contain four positive integer limits")
    if not config.get("reranker") or not config.get("embedding"):
        raise ValueError("Remote embedding AND reranker endpoints are required")


def remote_client(endpoint, config, cache):
    # The native scorer opens an httpx client internally. Bypass inherited proxies
    # for explicitly configured service hosts, just like the generation client.
    host = urlparse(endpoint["base_url"]).hostname
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(filter(None, [os.environ.get(key, ""), host]))
    client = AsyncOpenAI(base_url=endpoint["base_url"], api_key=os.getenv(endpoint.get("api_key_env", "VLLM_API_KEY"), "EMPTY"),
                         timeout=config["runtime"]["timeout_s"], max_retries=0,
                         http_client=httpx.AsyncClient(trust_env=False))
    return CachedAsyncOpenAI(client=client, rate_max_simultaneous=config["runtime"]["concurrency"],
                             retry_times_sec=(2, 4), cache=cache,
                             cache_prefix=digest(endpoint), embed_timeout=config["runtime"]["timeout_s"])


class Harness:
    def __init__(self, config, label):
        validate_config(config)
        self.config, self.protocol, self.label = config, config["protocol"], label
        endpoint = config["models"].get(label)
        if not endpoint:
            raise ValueError(f"Model endpoint is not configured: {label}")
        self.spec = manifest(config)
        self.root = ROOT / "runs" / ("ragu-" + digest(self.spec)[:16])
        self.folder = self.root / label
        write_json(self.root / "manifest.json", self.spec)
        endpoint_path = self.folder / "endpoint.json"
        if endpoint_path.exists() and read_json(endpoint_path) != endpoint:
            raise ValueError("Model endpoint changed within an existing run; use a new run/model label")
        write_json(endpoint_path, endpoint)
        cache = ROOT / "runs/cache/ragu"
        self.extract_llm = ProtocolLLM(endpoint, config, cache / "generation")
        self.answer_llm = ProtocolLLM(endpoint, config, cache / "generation", answer=True)
        self.embedding_client = remote_client(config["embedding"], config, cache / "embedding")
        self.scorer_client = remote_client(config["reranker"], config, cache / "reranker")
        self.embedder = EmbedderOpenAI(self.embedding_client, config["embedding"]["model"],
                                      dim=config["embedding"]["dimensions"], batch_size=16, max_concurrent_batches=2,
                                      embedder_token_limit=8192, tokenizer_backend="tiktoken", tokenizer_name="gpt-4o")
        self.reranker = ScorerOpenAI(self.scorer_client, config["reranker"]["model"])
        Settings.language = "russian"
        self.sparse = BM25(language="russian", cache_dir=str(ROOT / ".cache/fastembed"))

    def graph(self, user, *, smoke=False):
        Settings.storage_folder = str(self.folder / ("smoke" if smoke else "graphs") / user)
        extractor = TwoStageArtifactsExtractorLLM(
            llm=self.extract_llm, embedder=self.embedder,
            icl_config=ICLConfig(**self.protocol["icl"]), language="russian", ontology="nerel",
            do_entity_validation=self.protocol["entity_validation"],
            do_relation_validation=self.protocol["relation_validation"],
            show_type_signatures=True, prune_relation_types=True,
        )
        extractor.validator = AuditedValidator(extractor.validator)
        return KnowledgeGraph(
            llm=self.extract_llm, embedder=self.embedder, sparse_embedder=self.sparse, chunker=None,
            artifact_extractor=extractor, language="russian",
            builder_settings=BuilderArguments(**self.protocol["builder"]),
            storage_settings=StorageArguments(vdb_storage_type=SeparateQdrantStorage,
                                              vdb_storage_kwargs={"sparse_type": "bm25"}),
        )

    def engine(self, graph):
        return LocalSearchEngine(llm=self.answer_llm, knowledge_graph=graph, embedder=self.embedder,
                                 sparse_embedder=self.sparse, reranker=self.reranker, language="russian",
                                 max_context_length=self.protocol["context_max_tokens"],
                                 tokenizer_backend="tiktoken", tokenizer_model="gpt-4o")

    async def build(self, graph, user, sessions, *, smoke=False):
        documents, provenance = source_documents(user, sessions, self.protocol)
        storage = Path(Settings.storage_folder)
        write_json(storage / "chunk_sessions.json", provenance)
        started = time.monotonic()
        calls_before, errors_before = len(self.extract_llm.calls), len(self.extract_llm.errors)
        await graph.build_from_docs(documents)
        audit = await graph.index.check_consistency()
        nodes = await graph.index.graph_backend.get_all_nodes()
        edges = await graph.index.graph_backend.get_all_edges()
        stored_ids = set(await graph.index.chunks_kv_storage.all_keys())
        info = {"user": user, "sessions": len(sessions), "chunks": len(documents),
                "stored_chunks": len(stored_ids), "all_source_chunks_stored": set(provenance) <= stored_ids,
                "entities": len(nodes), "relations": len(edges), "consistent": audit.is_consistent,
                "consistency_errors": [asdict(issue) for issue in audit.errors],
                "seconds": time.monotonic() - started, "llm_calls": len(self.extract_llm.calls) - calls_before,
                "llm_errors": self.extract_llm.errors[errors_before:],
                "finish_reasons": [r["finish_reason"] for r in self.extract_llm.calls[calls_before:]],
                "ontology_reports": graph.pipeline.artifact_extractor.validator.reports}
        info["status"] = "complete_with_llm_errors" if info["llm_errors"] else "complete"
        write_json(storage / "build_report.json", info)
        if not audit.is_consistent or not nodes or not info["all_source_chunks_stored"]:
            raise ValueError(f"Invalid graph: {storage / 'build_report.json'}")
        print(canonical({"built": info}), flush=True)
        return provenance

    async def answer(self, engine, question, provenance, suite):
        path = self.folder / "answers" / suite / (question["id"].replace("|", "_") + ".json")
        if path.exists():
            return read_json(path)
        engine.update_prompt("local_search", answer_instruction(question))
        started = time.monotonic()
        response = await engine.query(question["question"], LocalParams(**self.protocol["search"]))
        raw = self.answer_llm.calls[-1]
        result = response.retrieval.result
        retrieved = {provenance[c.id] for c in result.chunks if c.id in provenance}
        prompt_text = "\n".join(m["content"] for m in raw["request"]["messages"])
        # Native context truncation can drop the tail of the source chunks.
        visible = {provenance[c.id] for c in result.chunks if c.id in provenance and c.content in prompt_text}
        evidence = set(question.get("evidence", []))
        row = {**question, "prediction": response.response, "seconds": time.monotonic() - started,
               "usage": raw["usage"], "finish_reason": raw["finish_reason"],
               "truncated": raw["finish_reason"] != "stop",
               "retrieved_sessions": sorted(retrieved), "visible_sessions": sorted(visible),
               "retrieval_recall": len(evidence & retrieved) / len(evidence) if evidence else None,
               "all_evidence_retrieved": evidence <= retrieved if evidence else None,
               "visible_evidence_recall": len(evidence & visible) / len(evidence) if evidence else None,
               "retrieval": {"entities": len(result.entities), "relations": len(result.relations), "chunks": len(result.chunks)},
               "raw_response_hash": digest(raw), "harness": self.spec["harness"]}
        write_json(path, row)
        write_json(self.folder / "retrieval" / suite / path.name,
                   {"question_id": question["id"], "context": response.retrieval.to_text(),
                    "messages": raw["request"]["messages"], "chunk_ids": [c.id for c in result.chunks]})
        print(canonical({"answered": question["id"], "prediction": row["prediction"],
                         "finish_reason": raw["finish_reason"], "retrieval": row["retrieval"]}), flush=True)
        return row

    async def close(self):
        self.extract_llm.close()
        self.answer_llm.close()
        await self.embedding_client.client.close()
        await self.scorer_client.client.close()


async def preflight(config, label):
    async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
        for name, endpoint in (("policy", config["models"][label]), ("embedding", config["embedding"]), ("reranker", config["reranker"])):
            response = await client.get(endpoint["base_url"].rstrip("/") + "/models")
            response.raise_for_status()
            names = [item["id"] for item in response.json()["data"]]
            if endpoint["model"] not in names:
                raise ValueError(f"{name}: expected {endpoint['model']}, available {names}")
            print(canonical({"service": name, "models": names}), flush=True)
        embedding = await client.post(config["embedding"]["base_url"].rstrip("/") + "/embeddings",
                                      json={"model": config["embedding"]["model"], "input": ["Пример личного факта"]})
        embedding.raise_for_status()
        if len(embedding.json()["data"][0]["embedding"]) != config["embedding"]["dimensions"]:
            raise ValueError("Embedding dimension mismatch")
        response = await client.post(config["reranker"]["base_url"].rstrip("/") + "/score",
                                     json={"model": config["reranker"]["model"], "text_1": "Красная планета",
                                           "text_2": ["Венера — планета.", "Марс называют Красной планетой."]})
        response.raise_for_status()
        scores = response.json()["data"]
        if max(scores, key=lambda r: r["score"])["index"] != 1:
            raise ValueError("Reranker smoke failed")
        print(canonical({"embedding_dimensions": config["embedding"]["dimensions"], "reranker": "ok"}), flush=True)


async def execute(args):
    config = read_json(args.config)
    validate_config(config)
    if args.command in ("score", "report"):
        if args.run_folder is None:
            raise ValueError("Provide --run-folder pointing to the saved model run")
        folder = args.run_folder.resolve()
        if not (folder.parent / "manifest.json").exists() or folder.name != args.model:
            raise ValueError("--run-folder must identify a saved RAGU model run")
        if args.command == "score":
            from .cli import score
            score(config, args.model, args.suite, folder_override=folder)
        else:
            make_report(folder, args.suite)
            print(comparison(folder.parent, args.suite, config["models"]))
        return
    if args.command == "preflight":
        await preflight(config, args.model)
        return
    harness = Harness(config, args.model)
    print(canonical({"run_folder": str(harness.folder)}), flush=True)
    try:
        if args.command == "smoke":
            sessions = [{"session_idx": 5, "session_datetime": "2025-03-05 10:00:00", "conversation": [
                {"role": "user", "message": "Сегодня я купила новую масляную пастель."},
                {"role": "assistant", "message": "Масляная пастель — материал для рисования. Храните её в коробке."}]},
                {"session_idx": 6, "session_datetime": "2025-03-09 11:00:00", "conversation": [
                    {"role": "user", "message": "А акварель я пока только выбираю, ещё не купила."},
                    {"role": "assistant", "message": "Можно посмотреть акварель марки Невская палитра."}]}]
            graph = harness.graph("smoke_user", smoke=True)
            try:
                provenance = await harness.build(graph, "smoke_user", sessions, smoke=True)
                row = await harness.answer(harness.engine(graph),
                    {"id": "smoke|q0", "user_id": "smoke_user", "question": "Когда я купила масляную пастель?",
                     "question_date": "2025-03-15 12:00:00", "answer": "5 марта 2025 года.",
                     "category": "StaticUser, single, temporal", "evidence": [5]}, provenance, "smoke")
                write_json(harness.folder / "smoke_result.json", row)
            finally:
                await graph.index.close()
            return
        ids = set(read_json(ROOT / "data/pilot50.json")) if args.suite == "pilot" else None
        questions = [q for q in read_jsonl(ROOT / "data/questions.jsonl") if ids is None or q["id"] in ids]
        if ids is not None and {q["id"] for q in questions} != ids:
            raise ValueError("Pilot question IDs do not match prepared dataset")
        write_json(harness.folder / (args.suite + "_question_ids.json"), [q["id"] for q in questions])
        for user in sorted({q["user_id"] for q in questions}):
            graph = harness.graph(user)
            try:
                report_path = Path(Settings.storage_folder) / "build_report.json"
                if not report_path.exists():
                    provenance = await harness.build(graph, user, read_json(ROOT / "data/conversations" / f"{user}.json"))
                else:
                    if not read_json(report_path)["consistent"]:
                        raise ValueError(f"Inconsistent graph: {report_path}")
                    provenance = read_json(Path(Settings.storage_folder) / "chunk_sessions.json")
                if args.command == "run":
                    engine = harness.engine(graph)
                    for question in (q for q in questions if q["user_id"] == user):
                        await harness.answer(engine, question, provenance, args.suite)
            finally:
                await graph.index.close()
        if args.command == "run":
            report = make_report(harness.folder, args.suite)
            rows = read_json(harness.folder / args.suite / "predictions.json")
            build_reports = [read_json(path) for path in (harness.folder / "graphs").glob("*/build_report.json")]
            write_json(harness.folder / args.suite / "qc.json", {
                "expected_questions": len(questions), "answered_questions": len(rows),
                "same_question_ids": {r["id"] for r in rows} == {q["id"] for q in questions},
                "truncated_answers": [r["id"] for r in rows if r.get("truncated")],
                "llm_extraction_errors": sum(len(r["llm_errors"]) for r in build_reports),
                "all_graphs_consistent": all(r["consistent"] for r in build_reports),
                "all_source_chunks_stored": all(r["all_source_chunks_stored"] for r in build_reports),
                "visible_evidence_recall": sum(r["visible_evidence_recall"] or 0 for r in rows) / len(rows),
                "judge_pending": report["overall"]["judged"] < len(questions),
            })
            comparison(harness.root, args.suite, config["models"])
            print(canonical({"report": report}), flush=True)
    finally:
        await harness.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "smoke", "index", "run", "score", "report"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.ragu.json")
    parser.add_argument("--model", default="meno-lite-0.2")
    parser.add_argument("--suite", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--run-folder", type=Path, help="Saved model folder, required for score/report")
    args = parser.parse_args()
    logger.remove()
    logger.add(lambda message: print(message, end="", flush=True), level="WARNING")
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
