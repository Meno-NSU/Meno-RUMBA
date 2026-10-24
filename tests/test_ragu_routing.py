"""The YaRN endpoint must never receive a request that fits the native budget."""
import asyncio
import copy

import pytest

pytest.importorskip("ragu")
from ragu.models.llm import LLM
from rumba_bench.ragu_routing import ContextRoutingLLM


class RecordingLLM(LLM):
    def __init__(self, name):
        self.name, self.calls, self.errors = name, [], []
        self.max_tokens = 4096

    async def chat_completion(self, conversation, output_schema=str, **kwargs):
        await asyncio.sleep(0)
        value = conversation[-1]["content"]
        self.calls.append({"text": value, "request": {"messages": conversation}, "finish_reason": "stop"})
        return self.name + ":" + value

    def close(self):
        pass


def router(tmp_path):
    native, extended = RecordingLLM("native"), RecordingLLM("yarn")
    config = {"runtime": {"concurrency": 4, "timeout_s": 2}, "protocol": {"context_routing": {
        "native_context_tokens": 32768, "extended_context_tokens": 65536}}}
    routed = ContextRoutingLLM(native, extended, {"model": "same", "base_url": "http://localhost/v1"}, config, tmp_path)
    routed.token_count = lambda conversation: int(conversation[-1]["content"])
    return routed, native, extended


@pytest.mark.asyncio
async def test_mixed_batch_and_exact_native_boundary(tmp_path):
    routed, native, extended = router(tmp_path)
    try:
        values = [128, 28672, 28673, 34640]
        conversations = [[{"role": "user", "content": str(value)}] for value in values]
        results = await routed.batch_chat_completion(conversations)
        assert results == ["native:128", "native:28672", "yarn:28673", "yarn:34640"]
        assert len(native.calls) == len(extended.calls) == 2
        assert all(raw["routing"]["budget_tokens"] <= 32768 for raw in routed.calls if raw["routing"]["backend"] == "native")
        assert all(raw["routing"]["budget_tokens"] > 32768 for raw in routed.calls if raw["routing"]["backend"] == "yarn")
    finally:
        routed.close()


@pytest.mark.asyncio
async def test_overflow_does_not_reach_either_model(tmp_path):
    routed, native, extended = router(tmp_path)
    try:
        with pytest.raises(ValueError, match="exceeds extended context"):
            await routed.chat_completion([{"role": "user", "content": "65536"}])
        assert not native.calls and not extended.calls and len(routed.errors) == 1
    finally:
        routed.close()


def test_exact_tokenizer_response_is_cached(tmp_path):
    import httpx
    routed, _, _ = router(tmp_path)
    del routed.token_count
    routed.tokenizer_client.close()
    requests = []

    def reply(request):
        requests.append(request)
        return httpx.Response(200, json={"count": 123, "tokens": [1, 2]})

    routed.tokenizer_client = httpx.Client(transport=httpx.MockTransport(reply))
    try:
        messages = [{"role": "user", "content": "Пример"}]
        assert routed.token_count(messages) == routed.token_count(copy.deepcopy(messages)) == 123
        assert len(requests) == 1 and requests[0].url.path == "/tokenize"
    finally:
        routed.close()


@pytest.mark.asyncio
async def test_tokenizer_retries_disconnects_without_changing_route(tmp_path, monkeypatch):
    import httpx
    from rumba_bench import ragu_routing

    routed, native, extended = router(tmp_path)
    del routed.token_count
    routed.tokenizer_client.close()
    requests, waits = [], []

    def reply(request):
        requests.append(request)
        if len(requests) <= 2:
            raise httpx.RemoteProtocolError("Server disconnected", request=request)
        return httpx.Response(200, json={"count": 34640})

    monkeypatch.setattr(ragu_routing.time, "sleep", waits.append)
    routed.tokenizer_client = httpx.Client(transport=httpx.MockTransport(reply))
    messages = [{"role": "user", "content": "34640"}]
    try:
        assert await routed.chat_completion(messages) == "yarn:34640"
        assert not native.calls and len(extended.calls) == 1
        assert len(requests) == 3 and waits == [1, 2]
        assert routed.token_count(messages) == 34640 and len(requests) == 3
        assert not routed.errors
    finally:
        routed.close()


@pytest.mark.asyncio
async def test_exhausted_tokenizer_retries_never_guess_a_route(tmp_path, monkeypatch):
    import httpx
    from rumba_bench import ragu_routing

    routed, native, extended = router(tmp_path)
    del routed.token_count
    routed.tokenizer_client.close()
    requests = []

    def reply(request):
        requests.append(request)
        raise httpx.ReadTimeout("Tokenizer unavailable", request=request)

    monkeypatch.setattr(ragu_routing.time, "sleep", lambda _: None)
    routed.tokenizer_client = httpx.Client(transport=httpx.MockTransport(reply))
    try:
        with pytest.raises(httpx.ReadTimeout):
            await routed.chat_completion([{"role": "user", "content": "34640"}])
        assert len(requests) == 3 and not list(tmp_path.glob("*.json"))
        assert not native.calls and not extended.calls and len(routed.errors) == 1
    finally:
        routed.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("native_limit,extended_limit,reranker_limit,message", [
    (32768, 65536, 512, "reranker: context 512 < required 8192"),
    (65536, 65536, 8192, "policy: context 65536 must equal native context 32768"),
    (32768, 32768, 8192, "policy_yarn: context 32768 must equal extended context 65536"),
])
async def test_preflight_rejects_incompatible_services_before_generation(
        tmp_path, monkeypatch, native_limit, extended_limit, reranker_limit, message):
    import httpx
    from rumba_bench import ragu_harness
    from rumba_bench.data import ROOT, read_json

    config = read_json(ROOT / "config.ragu.example.json")
    services = {8000: ("meno-lite-0.2", native_limit), 8001: ("bge-m3", 8192),
                8002: ("bge-reranker-v2-m3", reranker_limit), 8003: ("meno-lite-0.2", extended_limit)}
    requests = []

    def reply(request):
        requests.append(request)
        model, context = services[request.url.port]
        return httpx.Response(200, json={"data": [{"id": model, "max_model_len": context}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(ragu_harness.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(reply), **kwargs))
    monkeypatch.setattr(ragu_harness, "ROOT", tmp_path)
    with pytest.raises(ValueError, match=message):
        await ragu_harness.preflight(config, "meno-lite-0.2")
    assert len(requests) == 4 and all(request.method == "GET" for request in requests)


@pytest.mark.asyncio
async def test_deferred_judge_does_not_connect_to_inference_services(tmp_path, monkeypatch):
    from argparse import Namespace
    from rumba_bench import ragu_harness, cli
    from rumba_bench.data import ROOT, read_json, write_json

    config = read_json(ROOT / "config.ragu.example.json")
    config["long_context_models"] = {"meno-lite-0.2": None}
    path = tmp_path / "config.json"
    write_json(path, config)
    folder = tmp_path / "ragu-saved" / "meno-lite-0.2"
    folder.mkdir(parents=True)
    write_json(folder.parent / "manifest.json", {"harness": "saved"})
    calls = []

    async def forbidden(*args, **kwargs):
        raise AssertionError("Judge stage must not preflight the removed inference servers")

    monkeypatch.setattr(ragu_harness, "preflight", forbidden)
    monkeypatch.setattr(cli, "score", lambda config, label, suite, folder_override: calls.append(folder_override))
    await ragu_harness.execute(Namespace(config=path, command="score", model="meno-lite-0.2",
                                         suite="pilot", run_folder=folder))
    assert calls == [folder]
