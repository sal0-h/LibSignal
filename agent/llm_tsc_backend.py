"""Backends for the generic LLM traffic-signal controller."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence

from agent.llm_tsc_prompt import DEFAULT_API_FORMAT_SUFFIX


class LLMBackend:
    """Common interface: a message list produces normalized completion text."""

    def complete(self, messages: List[Dict[str, str]]) -> str:
        """Convenience wrapper around the canonical batch interface."""
        return self.complete_many([messages])[0]

    def complete_many(
        self,
        messages_list: Sequence[List[Dict[str, str]]],
    ) -> List[str]:
        """Complete requests in order; subclasses may provide concurrency."""
        return [self._complete_one(messages) for messages in messages_list]

    def _complete_one(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


def create_backend(param: Dict[str, Any]) -> LLMBackend:
    backend = str(param.get("backend", "api")).lower()
    if backend == "local":
        return LocalHFBackend(param)
    if backend == "api":
        return OpenAICompatibleBackend(param)
    raise ValueError(f"Unknown LLM backend {backend!r}; expected 'local' or 'api'")


class OpenAICompatibleBackend(LLMBackend):
    """Chat Completions against an OpenAI-compatible HTTP endpoint."""

    def __init__(self, param: Dict[str, Any]):
        self.base_url = str(param.get("base_url", "")).rstrip("/")
        if not self.base_url:
            raise ValueError(
                "model.base_url must be set in the agent YAML for backend=api"
            )
        self.model = str(param.get("model_id", "")).strip()
        if not self.model:
            raise ValueError(
                "model.model_id must be set in the agent YAML for backend=api"
            )

        key_env = str(param.get("api_key_env", "OPENAI_API_KEY"))
        self.api_key_env = key_env
        self.api_key = os.environ.get(key_env, "")
        if not self.api_key:
            raise ValueError(
                f"API key env var {key_env!r} is unset. Set it before running "
                "the API-backed LLM controller."
            )

        self.temperature = float(param.get("temperature", 0.0))
        self.max_tokens = int(param.get("max_new_tokens", 512))
        self.timeout = float(param.get("timeout", 120.0))
        self.extra_body = param.get("extra_body") or {}
        self.parallelism = max(1, int(param.get("parallelism", 16)))
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        from openai import OpenAI
                    except ImportError as exc:
                        raise ImportError(
                            "The 'openai' package is required for backend=api. "
                            "Install it with: pip install openai"
                        ) from exc
                    self._client = OpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key,
                        timeout=self.timeout,
                    )
        return self._client

    @staticmethod
    def _prepare_messages(
        messages: List[Dict[str, str]],
        suffix: str = DEFAULT_API_FORMAT_SUFFIX,
    ) -> List[Dict[str, str]]:
        """Copy messages and apply the API-only output-compliance suffix."""
        prepared = [dict(message) for message in messages]
        for message in reversed(prepared):
            if message.get("role") == "user":
                message["content"] = (
                    str(message.get("content", "")) + suffix
                )
                break
        return prepared

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(getattr(item, "text", item)))
            return "".join(parts)
        return str(content)

    def _extract_text(self, response: Any) -> str:
        if response is None:
            raise RuntimeError("LLM API returned an empty response")
        error = getattr(response, "error", None)
        if error:
            raise RuntimeError(f"LLM API error: {error}")
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("LLM API returned no choices")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError("LLM API returned a choice without a message")
        dump = message.model_dump() if hasattr(message, "model_dump") else {}
        content = self._message_text(message)
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or dump.get("reasoning_content")
            or dump.get("reasoning")
        )
        if reasoning and "<think>" not in content:
            content = f"<think>\n{reasoning}\n</think>\n{content}"
        return content

    def _complete_one(self, messages: List[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self._get_client().chat.completions.create(**kwargs)
        return self._extract_text(response)

    def complete_many(
        self,
        messages_list: Sequence[List[Dict[str, str]]],
    ) -> List[str]:
        if self.parallelism <= 1 or len(messages_list) <= 1:
            return [self._complete_one(messages) for messages in messages_list]

        with ThreadPoolExecutor(
            max_workers=min(self.parallelism, len(messages_list))
        ) as pool:
            futures = [pool.submit(self._complete_one, messages) for messages in messages_list]
            return [future.result() for future in futures]


class LocalHFBackend(LLMBackend):
    """Local transformers generation for the Traffic-R1 checkpoint."""

    def __init__(self, param: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for backend=local. "
                "Install with: pip install transformers accelerate"
            ) from exc

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

        print(
            f"[LLM] Loading local model from {model_path!r} "
            f"(subfolder={subfolder!r}) ..."
        )
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
        print("[LLM] Local model ready.")

    def _complete_one(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample and self.temperature > 0,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
