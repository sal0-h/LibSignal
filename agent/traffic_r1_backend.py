"""LLM backends for Traffic-R1: local HuggingFace and OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class TrafficR1Backend:
    """Common interface: messages -> raw completion text."""

    def complete(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


def create_backend(param: Dict[str, Any]) -> TrafficR1Backend:
    backend = str(param.get("backend", "api")).lower()
    if backend == "local":
        return LocalHFBackend(param)
    if backend == "api":
        return OpenAICompatibleBackend(param)
    raise ValueError(
        f"Unknown Traffic-R1 backend {backend!r}; expected 'local' or 'api'"
    )


class OpenAICompatibleBackend(TrafficR1Backend):
    """Chat Completions against an OpenAI-compatible HTTP endpoint."""

    def __init__(self, param: Dict[str, Any]):
        self.base_url = str(param.get("api_base_url", "")).rstrip("/")
        if not self.base_url:
            raise ValueError(
                "model.api_base_url must be set in the agent YAML for backend=api"
            )
        self.model = str(param.get("api_model", "")).strip()
        if not self.model:
            raise ValueError(
                "model.api_model must be set in the agent YAML for backend=api"
            )
        key_env = str(param.get("api_key_env", "OPENAI_API_KEY"))
        self.api_key = os.environ.get(key_env, "")
        if not self.api_key:
            raise ValueError(
                f"API key env var {key_env!r} is unset; export it before running "
                "Traffic-R1 with backend=api"
            )
        self.temperature = float(param.get("temperature", 0.0))
        self.max_tokens = int(param.get("max_new_tokens", 512))
        self.timeout = float(param.get("api_timeout", 120.0))
        # Optional OpenAI-compatible extras
        self.extra_body = param.get("api_extra_body") or {}

    def complete(self, messages: List[Dict[str, str]]) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for Traffic-R1 backend=api. "
                "Install with: pip install openai"
            ) from e

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        content = choice.content or ""
        # Some reasoning APIs put chain-of-thought in a separate field.
        reasoning = getattr(choice, "reasoning_content", None) or getattr(
            choice, "reasoning", None
        )
        if reasoning and "<think>" not in content:
            content = f"<think>\n{reasoning}\n</think>\n{content}"
        return content


class LocalHFBackend(TrafficR1Backend):
    """Local transformers generation for Season998/Traffic-R1 (Qwen2.5-3B)."""

    def __init__(self, param: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for Traffic-R1 backend=local. "
                "Install with: pip install transformers accelerate"
            ) from e

        self.torch = torch
        model_path = str(param.get("model_path", "Season998/Traffic-R1"))
        subfolder = param.get("model_subfolder", "huggingface")
        if subfolder is not None and str(subfolder).strip() == "":
            subfolder = None

        self.temperature = float(param.get("temperature", 0.0))
        self.max_new_tokens = int(param.get("max_new_tokens", 512))
        self.do_sample = bool(param.get("do_sample", False))

        load_kwargs: Dict[str, Any] = {}
        if subfolder:
            load_kwargs["subfolder"] = str(subfolder)

        device_map = param.get("device_map", "auto")
        torch_dtype = param.get("torch_dtype", "bfloat16")
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype == "float16":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = torch.float32

        print(f"[Traffic-R1] Loading local model from {model_path!r} "
              f"(subfolder={subfolder!r}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, **load_kwargs
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map if torch.cuda.is_available() else None,
            **load_kwargs,
        )
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()
        print("[Traffic-R1] Local model ready.")

    def complete(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample and self.temperature > 0,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
