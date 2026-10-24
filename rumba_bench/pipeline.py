"""Remote generation, local memory and retrieval. / Удалённая генерация, локальные память и поиск."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

import httpx
import numpy as np
from openai import OpenAI
from jinja2 import Template

from .data import ROOT, canonical, digest, read_json, timestamp, write_json
from .upstream import load_official
from .schemas import extraction_schema
from .episodic import citation_repair, fact_documents, fact_prompt, fact_schema, nerel_source, parse_facts, parse_repaired, windows

GREEDY_PARAMETERS = {"temperature": 0, "top_p": 1.0, "frequency_penalty": 0.0,
                     "presence_penalty": 0.0,
                     "extra_body": {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}}


def generation_parameters(settings=None):
    """Explicit request parameters; no checkpoint defaults. / Явные параметры генерации."""
    settings = settings or {}
    temperature = float(settings.get("temperature", 0))
    top_p = float(settings.get("top_p", 1))
    penalty = float(settings.get("repetition_penalty", 1))
    if not all(math.isfinite(x) for x in (temperature, top_p, penalty)) or temperature < 0 or not 0 < top_p <= 1 or penalty <= 0:
        raise ValueError("Invalid generation parameters")
    return {**GREEDY_PARAMETERS, "temperature": temperature, "top_p": top_p,
            "extra_body": {**GREEDY_PARAMETERS["extra_body"], "repetition_penalty": penalty}}


def training_modules(config):
    source = (ROOT / config["training_source"]).resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    # Load just the pure template submodule: loaders/__init__ imports training dependencies.
    # Загрузить только шаблоны: реестр загрузчиков тянет зависимости обучения.
    template_name = "llm_training.loaders._ragu_templates"
    if template_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(template_name, source / "llm_training/loaders/_ragu_templates.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[template_name] = module
        spec.loader.exec_module(module)
    from llm_training.extraction.ontology import get_ontology
    from llm_training.extraction.prompts import build_extraction_prompt
    from llm_training.extraction.retrieval import HybridIndex, RetrievalConfig
    return get_ontology("nerel"), build_extraction_prompt, HybridIndex, RetrievalConfig


def protocol_manifest(config):
    source = ROOT / config["training_source"] / "llm_training"
    files = ["extraction/ontology.py", "extraction/_nerel_bio_tables.py", "extraction/prompts.py",
             "extraction/retrieval.py", "loaders/_ragu_templates.py"]
    hashes = {name: digest((source / name).read_text()) for name in files}
    return {"protocol": config["protocol"], "generation_request_defaults": generation_parameters(config["protocol"]),
            "embedding": config["embedding"],
            "training_source_hashes": hashes,
            "runner_hashes": {p.name: digest(p.read_text()) for p in sorted((ROOT / "rumba_bench").glob("*.py"))},
            "upstream": read_json(ROOT / "vendor/rumba/manifest.json"),
            "dataset": read_json(ROOT / "data/manifest.json"),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("datasets", "openai", "numpy", "nltk", "jinja2")}}


def client_for(endpoint, runtime):
    return OpenAI(base_url=endpoint["base_url"], api_key=os.getenv(endpoint.get("api_key_env", "VLLM_API_KEY"), "EMPTY"),
                  timeout=runtime["timeout_s"], max_retries=runtime["max_retries"],
                  http_client=httpx.Client(trust_env=False))


class Generation:
    def __init__(self, endpoint, runtime, cache, sampling=None):
        self.endpoint, self.runtime, self.cache = endpoint, runtime, Path(cache)
        self.client = client_for(endpoint, runtime)
        self.sampling = generation_parameters(sampling)

    def call(self, messages, max_tokens, stop=None, seed=42, json_schema=None, allow_incomplete=False, structured_outputs=None):
        request = {"model": self.endpoint["model"], "messages": messages, **self.sampling,
                   "max_tokens": max_tokens, "seed": seed}
        if stop:
            request["stop"] = stop
        if json_schema is not None and structured_outputs is not None:
            raise ValueError("Use either json_schema or structured_outputs")
        constraint = {"json": json_schema} if json_schema is not None else structured_outputs
        if constraint is not None:
            request["extra_body"] = {**request["extra_body"], "structured_outputs": constraint}
        key = digest({"endpoint": self.endpoint, "request": request})
        path = self.cache / f"{key}.json"
        if path.exists():
            result = read_json(path)
        else:
            started = time.monotonic()
            response = self.client.chat.completions.create(**request)
            choice = response.choices[0]
            result = {"request": request, "text": choice.message.content or "",
                      "finish_reason": choice.finish_reason, "usage": response.usage.model_dump() if response.usage else {},
                      "seconds": time.monotonic() - started, "response_model": response.model,
                      "system_fingerprint": response.system_fingerprint}
            write_json(path, result)
        if result["finish_reason"] != "stop" and not allow_incomplete:
            raise ValueError(f"Generation incomplete ({result['finish_reason']}); raw response: {path}")
        return result


def chunks(text, size, overlap):
    if size <= overlap or overlap < 0:
        raise ValueError("chunk_chars must exceed nonnegative overlap")
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start + overlap:
                end = boundary
        yield text[start:end]
        if end == len(text):
            break
        start = end - overlap


def parse_extraction(text, ontology):
    """Parse JSON and record ontology violations. / Разобрать JSON и нарушения онтологии."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict) or not all(isinstance(parsed.get(key), list) for key in ("entities", "relations")):
        raise ValueError("Expected {entities: [...], relations: [...]}")
    entities, relations, violations, by_name = [], [], [], {}
    for entity in parsed["entities"]:
        if (not isinstance(entity, dict) or not all(isinstance(entity.get(k), str) and entity[k].strip()
            for k in ("entity_name", "entity_type", "description"))
            or not ontology.knows_entity_type(entity["entity_type"])):
            violations.append({"entity": entity})
            continue
        entities.append(entity)
        by_name[entity["entity_name"]] = entity["entity_type"]
    for relation in parsed["relations"]:
        valid = isinstance(relation, dict) and all(isinstance(relation.get(k), str) and relation[k].strip()
                  for k in ("source_entity", "target_entity", "relation_type", "description"))
        if valid:
            source, target = by_name.get(relation["source_entity"]), by_name.get(relation["target_entity"])
            strength = relation.get("relationship_strength")
            valid = (source is not None and target is not None and type(strength) is int and 0 <= strength <= 5
                     and ontology.allows(relation["relation_type"], source, target))
        if valid:
            relations.append(relation)
        else:
            violations.append({"relation": relation})
    return {"entities": entities, "relations": relations, "violations": violations}


class Encoder:
    def __init__(self, config, runtime):
        self.config = config
        self.local = None
        if config["backend"] == "openai":
            if not config.get("base_url"):
                raise ValueError("Configure embedding.base_url when the remote embedding server is ready")
            self.client = client_for(config, runtime)
        elif config["backend"] != "sentence-transformers":
            raise ValueError("Embedding backend must be openai or sentence-transformers")
        if config.get("revision") is None and config["backend"] == "sentence-transformers":
            raise ValueError("Pin embedding.revision to the HuggingFace commit before running")

    def encode(self, texts, query=False):
        prefix = self.config.get("query_prefix" if query else "passage_prefix", "")
        texts = [prefix + text for text in texts]
        if self.config["backend"] == "openai":
            rows = []
            for start in range(0, len(texts), self.config["batch_size"]):
                batch = texts[start:start + self.config["batch_size"]]
                response = self.client.embeddings.create(model=self.config["model"], input=batch)
                data = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in data] != list(range(len(batch))):
                    raise ValueError("Embedding response has missing/duplicate indices")
                rows.extend(item.embedding for item in data)
            return np.asarray(rows, dtype=np.float32)
        if self.local is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(self.config.get("threads", 4))
            self.local = SentenceTransformer(self.config["model"], revision=self.config["revision"],
                         device=self.config["device"], cache_folder=str(ROOT / ".cache/embeddings"))
        return self.local.encode(texts, batch_size=self.config["batch_size"], normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)

    def __call__(self, texts):
        return self.encode(texts, query=True)


def extract_user(user, config, generator, folder):
    if config["protocol"].get("memory_pipeline", "nerel_v1") == "episodic_v1":
        return extract_episodic_user(user, config, generator, folder)
    ontology, build_prompt, _, _ = training_modules(config)
    protocol = config["protocol"]
    schema = ontology.full_schema()
    output_schema = extraction_schema(ontology, protocol)
    sessions = sorted(read_json(ROOT / "data/conversations" / f"{user}.json"),
                      key=lambda s: (timestamp(s["session_datetime"]), s["session_idx"]))
    jobs = []
    for session in sessions:
        body = "\n".join(f"{m['role']}: {m['message']}" for m in session["conversation"])
        for part, text in enumerate(chunks(body, protocol["chunk_chars"], protocol["chunk_overlap_chars"])):
            source = f"User ID: {user}\nSession: {session['session_idx']}\nTimestamp: {session['session_datetime']}\n{text}"
            jobs.append((session, part, source))

    progress = {"done": 0, "errors": 0}
    lock = threading.Lock()

    def process(job):
        session, part, source = job
        path = folder / "extractions" / user / f"{session['session_idx']}-{part}.json"
        if path.exists():
            record = read_json(path)
        else:
            messages = build_prompt("one_stage", source, schema, language="Russian",
                                   variant=protocol["prompt_variant"], with_signatures=protocol["with_signatures"])
            record = {"session_idx": session["session_idx"], "timestamp": session["session_datetime"],
                      "chunk": part, "source_sha256": digest(source)}
            try:
                response = generator.call(messages, protocol["extraction_max_tokens"], seed=protocol["seed"],
                                          json_schema=output_schema)
                record.update(parse_extraction(response["text"], ontology))
                record["seconds"] = response["seconds"]
                record["usage"] = response["usage"]
            except (ValueError, json.JSONDecodeError) as error:
                record.update({"error": str(error), "entities": [], "relations": [], "violations": []})
            write_json(path, record)
        with lock:
            progress["done"] += 1
            progress["errors"] += int("error" in record)
            if progress["done"] % 10 == 0 or progress["done"] == len(jobs):
                print(f"{user}: {progress['done']}/{len(jobs)} chunks; {progress['errors']} errors / ошибок", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=config["runtime"]["concurrency"]) as pool:
        records = list(pool.map(process, jobs))
    documents = []
    for record in records:
        for kind in ("entities", "relations"):
            for i, artifact in enumerate(record[kind]):
                documents.append({"id": f"{user}/{record['session_idx']}/{record['chunk']}/{kind}/{i}",
                                  "session_idx": record["session_idx"], "timestamp": record["timestamp"],
                                  "text": canonical(artifact)})
    write_json(folder / "memories" / f"{user}.json", {"records": records, "documents": documents})
    print(f"{user}: {len(jobs)} chunks, {len(documents)} memories, "
          f"{sum('error' in r for r in records)} errors / ошибок", flush=True)
    return records, documents


def extract_memory_window(user, session, part, segments, config, generator, folder):
    ontology, build_prompt, _, _ = training_modules(config)
    protocol = config["protocol"]
    path = folder / "extractions" / user / f"{session['session_idx']}-{part}.json"
    if path.exists():
        return read_json(path)
    record = {"session_idx": session["session_idx"], "timestamp": session["session_datetime"],
              "chunk": part, "source_sha256": digest(segments), "facts": [], "rejected_facts": [],
              "entities": [], "relations": [], "violations": []}
    try:
        response = generator.call(fact_prompt(user, session, segments), protocol.get("memory_fact_max_tokens", 4096),
                                  seed=protocol["seed"], json_schema=fact_schema(protocol) if protocol["extraction_decoding"] == "structured" else None)
        facts, rejected = parse_facts(response["text"], segments)
        record.update({"facts": facts, "rejected_facts": rejected, "seconds": response["seconds"], "usage": response["usage"]})
    except ValueError as error:
        record["error"] = str(error)
        write_json(path, record)
        return record
    if record["rejected_facts"] and protocol.get("memory_citation_repair", True):
        repair = citation_repair(record["rejected_facts"], segments, protocol)
        if repair is not None:
            candidates, messages, schema = repair
            try:
                response = generator.call(messages, protocol.get("memory_repair_max_tokens", 3072), seed=protocol["seed"],
                                          json_schema=schema if protocol["extraction_decoding"] == "structured" else None)
                repaired = parse_repaired(response["text"], candidates, segments)
                record["repaired_facts"] = repaired
                record["repair_usage"] = response["usage"]
                record["repair_seconds"] = response["seconds"]
                seen = {digest(f) for f in record["facts"]}
                record["facts"].extend(f for f in repaired if digest(f) not in seen)
            except ValueError as error:
                record["repair_error"] = str(error)
    if record["facts"]:
        messages = build_prompt("one_stage", nerel_source(record["facts"]), ontology.full_schema(), language="Russian",
                                variant=protocol["prompt_variant"], with_signatures=protocol["with_signatures"])
        try:
            response = generator.call(messages, protocol["extraction_max_tokens"], seed=protocol["seed"],
                                      json_schema=extraction_schema(ontology, protocol))
            record.update(parse_extraction(response["text"], ontology))
            record["nerel_usage"] = response["usage"]
            record["nerel_seconds"] = response["seconds"]
        except ValueError as error:
            # A graph annotation error must not erase already verified personal facts.
            # Ошибка разметки графа не должна стирать уже проверенные личные факты.
            record["nerel_error"] = str(error)
    write_json(path, record)
    return record


def extract_episodic_user(user, config, generator, folder):
    sessions = sorted(read_json(ROOT / "data/conversations" / f"{user}.json"),
                      key=lambda s: (timestamp(s["session_datetime"]), s["session_idx"]))
    jobs = [(session, part, segments) for session in sessions
            for part, segments in enumerate(windows(session, chunks, config["protocol"].get("memory_window_chars", 8000)))]
    records = []
    def process(job):
        session, part, segments = job
        return extract_memory_window(user, session, part, segments, config, generator, folder)
    with ThreadPoolExecutor(max_workers=config["runtime"]["concurrency"]) as pool:
        for record in pool.map(process, jobs):
            records.append(record)
            if len(records) % 10 == 0 or len(records) == len(jobs):
                print(f"{user}: {len(records)}/{len(jobs)} windows; {sum('error' in r for r in records)} fact errors / ошибок фактов", flush=True)
    documents = fact_documents(user, records)
    write_json(folder / "memories" / f"{user}.json", {"records": records, "documents": documents})
    print(f"{user}: {len(records)} windows, {len(documents)} grounded memories / подтверждённых фактов", flush=True)
    return records, documents


def index_user(user, documents, config, encoder, folder):
    _, _, HybridIndex, RetrievalConfig = training_modules(config)
    directory = folder / "indexes" / user
    if not documents:
        return None
    if (directory / "snapshot.json").exists():
        return HybridIndex.load(directory, encode=encoder)
    texts = [f"{d['timestamp']}: {d.get('index_text', d['text'])}" for d in documents]
    vectors = encoder.encode(texts)
    index = HybridIndex.build(texts=texts, doc_ids=[d["id"] for d in documents], encode=encoder,
                              config=RetrievalConfig(candidate_pool=config["protocol"]["candidate_pool"]), vectors=vectors)
    write_json(directory / "digests.json", index.save(directory))
    return index


def retrieve_question(question, documents, index, config, official):
    protocol = config["protocol"]
    hits = index.search(question["question"], protocol["top_k"]) if index is not None else []
    by_id = {d["id"]: d for d in documents}
    memories, selected, size, picked = [], [], 0, []
    for hit in hits:
        doc = by_id[hit.doc_id]
        date = timestamp(doc["timestamp"])
        text = f"{doc['timestamp']} ({date.strftime('%A')}): {doc['text']}"
        if size + len(text) > protocol["context_max_chars"]:
            continue
        size += len(text)
        picked.append((doc, text))
        selected.append({**asdict(hit), "session_idx": doc["session_idx"]})
    if protocol.get("memory_pipeline", "nerel_v1") == "episodic_v1":
        picked.sort(key=lambda item: (timestamp(item[0]["timestamp"]), item[0]["session_idx"], item[0].get("source_turn", 0), item[0]["id"]))
    memories = [text for _, text in picked]
    date = timestamp(question["question_date"])
    prompt = Template(official["ANSWER_PROMPT_RU"]).render(memories=memories, question=question["question"],
                                                         query_date=date, query_weekday=date.strftime("%A"))
    evidence = set(question["evidence"])
    retrieved = {doc["session_idx"] for doc in selected}
    return {"id": question["id"], "prompt": prompt, "retrieved": selected,
            "retrieval_recall": len(evidence & retrieved) / len(evidence) if evidence else None,
            "all_evidence_retrieved": evidence <= retrieved if evidence else None,
            "context_chars": size, "omitted_by_budget": len(hits) - len(selected)}


def generate_answer(question, retrieved, config, generator):
    protocol = config["protocol"]
    response = generator.call([{"role": "user", "content": retrieved["prompt"]}], protocol["answer_max_tokens"],
                              stop=protocol["answer_stop"], seed=protocol["seed"])
    return {**question, **{k: v for k, v in retrieved.items() if k != "prompt"}, "prediction": response["text"],
            "seconds": response["seconds"], "usage": response["usage"]}


def answer_question(question, documents, index, config, generator, official):
    return generate_answer(question, retrieve_question(question, documents, index, config, official), config, generator)
