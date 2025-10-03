import os
import asyncio
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import aiohttp

from transformers import AutoTokenizer


class TokenizerCache:
    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def get(self, model_name: str):
        if model_name not in self._cache:
            # Do NOT rewrite model name; try to load as-is. If fails, cache None and fallback later.
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except Exception:
                tokenizer = None
            self._cache[model_name] = tokenizer
        return self._cache[model_name]


def build_app() -> FastAPI:
    app = FastAPI()
    tokenizer_cache = TokenizerCache()

    backend_address = os.environ.get("VLLM_BACKEND_ADDRESS", "127.0.0.1:8101")
    # Handle address that might already contain protocol
    if backend_address.startswith(('http://', 'https://')):
        backend_base = f"{backend_address}/v1"
    else:
        backend_base = f"http://{backend_address}/v1"

    timeout = aiohttp.ClientTimeout(total=2700)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{backend_base}/models") as resp:
                data = await resp.json()
                return JSONResponse(content=data, status_code=resp.status)

    @app.post("/v1/completions")
    async def completions(request: Request):
        req_json = await request.json()

        model_name = req_json.get("model")
        if not model_name:
            return JSONResponse(content={"error": "model is required"}, status_code=400)

        # Keep model name as-is to match vLLM served-model-name exactly
        req_json_copy = req_json.copy()

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{backend_base}/completions", json=req_json_copy) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return JSONResponse(content=data, status_code=resp.status)

        try:
            choices = data.get("choices", [])
            if not choices:
                return JSONResponse(content=data)

            tokenizer = tokenizer_cache.get(model_name)

            for choice in choices:
                logprobs = choice.get("logprobs")
                if not logprobs:
                    continue
                tokens = logprobs.get("tokens", [])
                if not tokens:
                    continue

                token_ids = []
                if tokenizer is None:
                    # Fallback: preserve format with unknown id
                    token_ids = ["token_id:-1" for _ in tokens]
                else:
                    for tok in tokens:
                        try:
                            tid = tokenizer.convert_tokens_to_ids(tok)
                            if isinstance(tid, int) and tid >= 0:
                                token_ids.append(f"token_id:{tid}")
                            else:
                                token_ids.append("token_id:-1")
                        except Exception:
                            token_ids.append("token_id:-1")

                logprobs["tokens"] = token_ids

        except Exception:
            pass

        return JSONResponse(content=data)

    return app


def main() -> None:
    host = os.environ.get("PROXY_HOST", "127.0.0.1")
    port = int(os.environ.get("PROXY_PORT", "8100"))
    app = build_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()




