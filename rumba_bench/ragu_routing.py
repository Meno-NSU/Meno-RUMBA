"""Route short requests to native RoPE and long requests to a separate YaRN server."""
import asyncio
import os
from pathlib import Path
import time

import httpx
from ragu.models.llm import LLM

from .data import digest, read_json, write_json


class ContextRoutingLLM(LLM):
    """Use exact remote token counts; never send a native-sized request to YaRN.

    The budget includes the reserved output tokens. The two backends must load
    the same weights and chat template; only the long backend enables YaRN.
    """

    def __init__(self, native, extended, native_endpoint, config, cache):
        self.native, self.extended = native, extended
        self.native_endpoint = native_endpoint
        self.settings = config["protocol"]["context_routing"]
        self.native_limit = self.settings["native_context_tokens"]
        self.extended_limit = self.settings["extended_context_tokens"]
        if not 0 < self.native_limit < self.extended_limit:
            raise ValueError("Native context must be positive and smaller than extended context")
        if native.max_tokens != extended.max_tokens:
            raise ValueError("Both backends must use identical generation limits")
        self.max_tokens = native.max_tokens
        self.cache = Path(cache)
        self.tokenizer_max_retries = config["runtime"].get("max_retries", 2)
        if type(self.tokenizer_max_retries) is not int or self.tokenizer_max_retries < 0:
            raise ValueError("runtime.max_retries must be a nonnegative integer")
        key = os.getenv(native_endpoint.get("api_key_env", "VLLM_API_KEY"), "EMPTY")
        self.tokenizer_client = httpx.Client(trust_env=False, timeout=config["runtime"]["timeout_s"],
                                           headers={"Authorization": f"Bearer {key}"})
        self.semaphore = asyncio.Semaphore(config["runtime"]["concurrency"])
        self.calls, self.errors = [], []

    def token_count(self, conversation):
        request = {"model": self.native_endpoint["model"], "messages": conversation, "add_generation_prompt": True}
        key = digest({"endpoint": self.native_endpoint, "request": request})
        path = self.cache / (key + ".json")
        if path.exists():
            count = read_json(path)["count"]
        else:
            base = self.native_endpoint["base_url"].rstrip("/").removesuffix("/v1")
            # Tokenization is safe to repeat. A transient keep-alive disconnect
            # must not turn a chunk into a missing extraction result upstream.
            for attempt in range(self.tokenizer_max_retries + 1):
                try:
                    response = self.tokenizer_client.post(base + "/tokenize", json=request)
                    response.raise_for_status()
                    break
                except httpx.TransportError:
                    if attempt == self.tokenizer_max_retries:
                        raise
                    time.sleep(min(2 ** attempt, 4))
            count = response.json()["count"]
            write_json(path, {"count": count, "request_hash": digest(request)})
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid tokenizer count: {count!r}")
        return count

    async def chat_completion(self, conversation, output_schema=str, **kwargs):
        async with self.semaphore:
            started = time.monotonic()
            routing = {}
            try:
                count = await asyncio.to_thread(self.token_count, conversation)
                budget = count + self.max_tokens
                if budget <= self.native_limit:
                    backend, route, limit = self.native, "native", self.native_limit
                elif budget <= self.extended_limit:
                    backend, route, limit = self.extended, "yarn", self.extended_limit
                else:
                    raise ValueError(f"Request budget {budget} exceeds extended context {self.extended_limit}")
                routing = {"backend": route, "prompt_tokens": count, "reserved_output_tokens": self.max_tokens,
                           "budget_tokens": budget, "context_tokens": limit}
                result = await backend.chat_completion(conversation, output_schema=output_schema, **kwargs)
                raw = backend.calls[-1]
                self.calls.append({**raw, "routing": routing, "routing_elapsed_seconds": time.monotonic() - started})
                return result
            except Exception as error:
                self.errors.append({"error": str(error), "conversation_hash": digest(conversation),
                                    "schema": output_schema.__name__, "routing": routing})
                raise

    def close(self):
        self.native.close()
        self.extended.close()
        self.tokenizer_client.close()
