"""Offline integration checks for the real upstream RAGU, without model weights."""
import json

import httpx
import numpy as np
from openai import OpenAI
import pytest
from pydantic import BaseModel

pytest.importorskip("ragu")
from ragu.common.global_parameters import Settings
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.index import StorageArguments
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.models.scorer import Scorer
from ragu.models.sparse_embedder import SparseEmbedder
from ragu.search_engine.local_search import LocalParams, LocalSearchEngine
from ragu.storage.types import SparseEmbedding

from rumba_bench.data import ROOT, read_json
from rumba_bench.ragu_harness import (AuditedValidator, ProtocolLLM, SeparateQdrantStorage,
                                      answer_instruction, generation_schema, source_documents)


class TinyEmbedder(Embedder):
    @property
    def dim(self):
        return 3

    async def embed_text(self, text, **kwargs):
        return np.array([1.0, 0.0, 0.0])

    async def embed_texts(self, texts, **kwargs):
        return np.array([[1.0, 0.0, 0.0] for _ in texts])


class TinySparse(SparseEmbedder):
    def embed_document(self, texts):
        return [SparseEmbedding(indices=[1], values=[1.0]) for _ in texts]

    def embed_query(self, texts):
        return self.embed_document(texts)


class FixedLLM(LLM):
    async def chat_completion(self, conversation, output_schema=str, **kwargs):
        return "5 марта 2025 года."


class RecordingScorer(Scorer):
    def __init__(self):
        self.calls = []

    async def score(self, text_1, text_2, **kwargs):
        self.calls.append((text_1, text_2))
        return [(i, 1.0) for i in range(len(text_2))]


def test_source_metadata_repeated_and_no_question_leak():
    config = read_json(ROOT / "config.ragu.example.json")
    sessions = [{"session_idx": 5, "session_datetime": "2025-03-05 10:00:00", "conversation": [
        {"role": "user", "message": "Купила масляную пастель."},
        {"role": "assistant", "message": "Общий совет. " * 1000}]}]
    docs, provenance = source_documents("user_test", sessions, config["protocol"])
    assert len(docs) > 1
    assert len(provenance) == len(docs)
    assert set(provenance.values()) == {5}
    assert all("Session datetime: 2025-03-05 10:00:00" in doc and "User ID: user_test" in doc for doc in docs)
    assert "user: Купила масляную пастель." in docs[0]
    assert all("Fragment starts in role: assistant" in doc for doc in docs[1:])
    question = {"question_date": "2025-03-15 12:00:00", "answer": "SECRET_GOLD", "evidence": [999]}
    from ragu.common.prompts.messages import render
    rendered = render(answer_instruction(question).messages, query="Когда?", context="FACT", language="russian")
    assert "SECRET_GOLD" not in str(rendered)


@pytest.mark.asyncio
async def test_transport_forwards_protocol_and_keeps_truncation(tmp_path):
    config = read_json(ROOT / "config.ragu.example.json")
    adapter = ProtocolLLM({"model": "test", "base_url": "http://localhost/v1"}, config, tmp_path, answer=True)
    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0, "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "raw repeated text"}, "finish_reason": "length"}]})

    adapter.generator.client.close()
    adapter.generator.client = OpenAI(base_url="http://localhost/v1", api_key="EMPTY",
                                       http_client=httpx.Client(transport=httpx.MockTransport(reply)))
    result = await adapter.chat_completion([{"role": "user", "content": "Q"}])
    assert result == "raw repeated text"
    assert adapter.calls[-1]["finish_reason"] == "length"
    assert requests[0]["temperature"] == 0
    assert requests[0]["seed"] == 42
    assert requests[0]["repetition_penalty"] == 1.03
    assert requests[0]["top_k"] == -1 and requests[0]["min_p"] == 0
    with pytest.raises(ValueError, match="overrides"):
        await adapter.chat_completion([], temperature=0.5)
    adapter.close()


@pytest.mark.asyncio
async def test_incomplete_structured_extraction_is_an_audited_error(tmp_path):
    class Artifacts(BaseModel):
        entities: list[str]

    config = read_json(ROOT / "config.ragu.example.json")
    adapter = ProtocolLLM({"model": "test", "base_url": "http://localhost/v1"}, config, tmp_path)
    adapter.generator.client.close()
    adapter.generator.call = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Generation incomplete (length)"))
    with pytest.raises(ValueError, match="incomplete"):
        await adapter.chat_completion([{"role": "user", "content": "SOURCE"}], output_schema=Artifacts)
    assert len(adapter.errors) == 1 and not adapter.calls
    assert len(list((tmp_path / "errors").glob("*.json"))) == 1


def test_ontology_audit_preserves_native_decisions_and_is_json_serializable():
    from ragu.common.prompts.default_models import EntityModel, RelationModel
    from ragu.triplet.ontology import OntologyValidator
    native = OntologyValidator("nerel")
    audit = AuditedValidator(native)
    entities = [[EntityModel(entity_name="Юзер", entity_type="PERSON", description="Юзер.")]]
    relations = [[RelationModel(source_entity="Юзер", target_entity="Юзер", relation_type="PART_OF",
                                description="Некорректная петля.", relationship_strength=1)]]
    expected = native.validate_batch(entities, relations)
    actual = audit.validate_batch(entities, relations)
    assert actual[:2] == expected[:2]
    assert "self_loop" in json.dumps(audit.reports)


def test_bounded_native_and_free_modes_preserve_upstream_contract():
    from ragu.common.prompts.default_models import EntitiesExtractionModel, RelationsExtractionModel
    protocol = read_json(ROOT / "config.ragu.example.json")["protocol"]
    for model, field in ((EntitiesExtractionModel, "entities"), (RelationsExtractionModel, "relations")):
        schema = generation_schema(model, protocol)
        assert schema["properties"][field]["maxItems"] == 16
        definition = next(iter(schema["$defs"].values()))
        assert definition["properties"]["description"]["maxLength"] == 200
        assert "maxItems" not in model.model_json_schema()["properties"][field]
        assert generation_schema(model, {**protocol, "extraction_decoding": "native"}) == model.model_json_schema()
        assert generation_schema(model, {**protocol, "extraction_decoding": "free"}) is None


@pytest.mark.asyncio
async def test_hybrid_collections_persist_and_do_not_cross_users(tmp_path, monkeypatch):
    from ragu.chunker.types import Chunk
    from ragu.graph.types import Entity
    import tiktoken

    class CharacterCodec:
        def encode(self, text):
            return list(text)

        def decode(self, tokens):
            return "".join(tokens)

    # No tokenizer downloads in this storage/retrieval integration test.
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda _: CharacterCodec())

    embedder, sparse = TinyEmbedder(), TinySparse()
    scorer = RecordingScorer()

    def open_graph(user):
        Settings.storage_folder = str(tmp_path / user)
        return KnowledgeGraph(llm=FixedLLM(), embedder=embedder, sparse_embedder=sparse,
            builder_settings=BuilderArguments(build_only_vector_context=True, make_community_summary=False),
            storage_settings=StorageArguments(vdb_storage_type=SeparateQdrantStorage,
                                             vdb_storage_kwargs={"sparse_type": "bm25"}))

    graph = open_graph("user_a")
    chunk = Chunk("user: Купила пастель 5 марта 2025 года.", 0, "doc_0")
    entity = Entity(entity_name="Пастель", entity_type="PRODUCT", description="Личная покупка пользователя.",
                    source_chunk_id=[chunk.id], documents_id=["doc_0"])
    try:
        await graph.index.upsert_chunks([chunk])
        await graph.index.upsert_nodes([entity])
    finally:
        await graph.index.close()
    graph = open_graph("user_a")
    try:
        engine = LocalSearchEngine(FixedLLM(), graph, embedder, sparse, scorer,
                                   max_context_length=1000, tokenizer_backend="tiktoken", tokenizer_model="gpt-4o")
        response = await engine.query("Когда купила пастель?", LocalParams(top_k=2, rerank_top_k=1, use_chunks=True))
        assert response.retrieval.result.chunks[0].content == chunk.content
        assert scorer.calls
        assert (await graph.index.check_consistency()).is_consistent
    finally:
        await graph.index.close()
    graph = open_graph("user_b")
    try:
        assert await graph.index.graph_backend.get_all_nodes() == []
        assert await graph.index.chunks_kv_storage.all_keys() == []
    finally:
        await graph.index.close()
