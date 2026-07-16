"""Pluggable LLM backend with unified interface + cost accounting.

The adaptive controller's whole contribution is *reducing LLM/token cost*, so
every backend call is instrumented to report (n_input_tokens, n_output_tokens,
n_calls). Two backends share the same interface:

  - HuggingFaceBackend : local open-weight model on the single GPU (default,
    fully reproducible, zero API spend). Tested with 7-8B models on a 32GB
    RTX 5090 in bf16.
  - APIBackend          : OpenAI-compatible chat completion endpoint (optional;
    set base_url / api_key for any compatible provider). Disabled by default.

Both expose `generate(prompt, max_new_tokens)` -> dict with 'text' and 'usage'.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass
class Usage:
    n_calls: int = 0
    n_input_tokens: int = 0
    n_output_tokens: int = 0

    def add(self, inp: int, out: int) -> None:
        self.n_calls += 1
        self.n_input_tokens += int(inp)
        self.n_output_tokens += int(out)

    def total_tokens(self) -> int:
        return self.n_input_tokens + self.n_output_tokens

    def as_dict(self) -> Dict[str, int]:
        return {
            "n_calls": self.n_calls,
            "n_input_tokens": self.n_input_tokens,
            "n_output_tokens": self.n_output_tokens,
            "total_tokens": self.total_tokens(),
        }


class LLMBackend(Protocol):
    usage: Usage

    def generate(self, prompt: str, max_new_tokens: int = 64,
                 temperature: float = 0.0, stop: Optional[List[str]] = None) -> Dict: ...

    def embed(self, texts: List[str]) -> "List[List[float]]": ...


# --------------------------------------------------------------------------- HF
class HuggingFaceBackend:
    """Local open-weight causal LM + a sentence-transformer for embeddings.

    Lazy-loaded so importing the module is cheap. The embedder is shared across
    retrieval; the generator handles answer + relevance-judge calls.
    """

    def __init__(
        self,
        gen_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.gen_model_name = gen_model
        self.embed_model_name = embed_model
        self.device = device
        self.dtype = dtype
        self.usage = Usage()
        self._gen = None  # lazy
        self._tok = None
        self._embed = None

    def _ensure_gen(self):
        if self._gen is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = getattr(torch, self.dtype, torch.bfloat16)
        self._tok = AutoTokenizer.from_pretrained(self.gen_model_name)
        self._gen = AutoModelForCausalLM.from_pretrained(
            self.gen_model_name, torch_dtype=dtype, device_map=self.device
        )
        self._gen.eval()
        if self._tok.pad_token_id is None:
            self._tok.pad_token_id = self._tok.eos_token_id

    def _ensure_embed(self):
        if self._embed is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._embed = SentenceTransformer(self.embed_model_name, device=self.device)

    def generate(self, prompt: str, max_new_tokens: int = 64,
                 temperature: float = 0.0, stop: Optional[List[str]] = None) -> Dict:
        self._ensure_gen()
        import torch
        inputs = self._tok(prompt, return_tensors="pt").to(self._gen.device)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            do_sample = temperature > 0
            out = self._gen.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5) if do_sample else 1.0,
                pad_token_id=self._tok.pad_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = self._tok.decode(gen_ids, skip_special_tokens=True)
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        n_out = int(gen_ids.shape[0])
        self.usage.add(n_in, n_out)
        return {"text": text.strip(), "usage": {"n_input_tokens": n_in, "n_output_tokens": n_out}}

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_embed()
        if not texts:
            return []
        import numpy as np
        emb = self._embed.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)


# --------------------------------------------------------------------------- API
class APIBackend:
    """OpenAI-compatible chat backend (optional).

    Disabled unless AGR_API_KEY is set. Useful for ablations with frontier
    models; local HF remains the default for full reproducibility.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ):
        self.model = model
        self.base_url = base_url or os.environ.get("AGR_API_BASE")
        self.api_key = api_key or os.environ.get("AGR_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "APIBackend requires AGR_API_KEY (and optionally AGR_API_BASE)."
            )
        self.usage = Usage()
        self._embed_model = embed_model
        self._device = device
        self._embed = None
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _ensure_embed(self):
        if self._embed is None:
            from sentence_transformers import SentenceTransformer
            self._embed = SentenceTransformer(self._embed_model, device=self._device)

    def generate(self, prompt: str, max_new_tokens: int = 64,
                 temperature: float = 0.0, stop: Optional[List[str]] = None) -> Dict:
        self._ensure_client()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        self.usage.add(u.prompt_tokens, u.completion_tokens)
        return {"text": text.strip(), "usage": {"n_input_tokens": u.prompt_tokens,
                                                 "n_output_tokens": u.completion_tokens}}

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_embed()
        if not texts:
            return []
        emb = self._embed.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)


def build_backend(cfg: Dict) -> LLMBackend:
    """Factory from a config dict. Defaults to local HF (no API spend)."""
    kind = cfg.get("kind", "hf")
    if kind == "api":
        return APIBackend(model=cfg.get("model", "gpt-4o-mini"),
                          base_url=cfg.get("base_url"), api_key=cfg.get("api_key"),
                          embed_model=cfg.get("embed_model",
                                              "sentence-transformers/all-MiniLM-L6-v2"))
    return HuggingFaceBackend(
        gen_model=cfg.get("gen_model", "Qwen/Qwen2.5-1.5B-Instruct"),
        embed_model=cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", "bfloat16"),
    )
