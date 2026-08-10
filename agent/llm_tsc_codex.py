"""Small, direct OpenAI Codex Responses adapter for the LLM TSC agent.

Hermes calls the ChatGPT Codex endpoint through the OpenAI Responses wire
format (``codex_responses``).  This module ports only the pieces needed by
LibSignal's one-shot traffic decisions.  It intentionally does not invoke
the Codex CLI or app-server runtime, and it does not implement OAuth token
refresh or write to either auth store.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.llm_tsc_prompt import DEFAULT_API_FORMAT_SUFFIX

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_MODEL = "gpt-5.6-luna"


class CodexAuthError(RuntimeError):
    """Raised when a supported local Codex credential source is unavailable."""


@dataclass(frozen=True)
class CodexCredentials:
    """In-memory credentials resolved from a supported local auth file.

    ``repr=False`` prevents an accidental debug/log representation from
    exposing the bearer token.  The token is never written by this adapter.
    """

    access_token: str = field(repr=False)
    source: str
    base_url: str = DEFAULT_CODEX_BASE_URL


def _path_from_param(param: Mapping[str, Any], key: str, default: Path) -> Path:
    value = param.get(key)
    return Path(str(value)).expanduser() if value else default


def _supported_auth_paths(param: Mapping[str, Any]) -> Tuple[Path, Path]:
    """Return Hermes and Codex CLI auth paths without reading their contents."""
    hermes_home = (
        Path(os.environ["HERMES_HOME"]).expanduser()
        if os.environ.get("HERMES_HOME", "").strip()
        else Path.home() / ".hermes"
    )
    codex_home = (
        Path(os.environ["CODEX_HOME"]).expanduser()
        if os.environ.get("CODEX_HOME", "").strip()
        else Path.home() / ".codex"
    )

    # ``auth_path`` is a convenient explicit override for tests and for a
    # Hermes profile whose path is already known.  The provider-specific
    # names remain available when both stores need to be configured.
    hermes_path = _path_from_param(
        param,
        "hermes_auth_path",
        _path_from_param(param, "auth_path", hermes_home / "auth.json"),
    )
    codex_path = _path_from_param(param, "codex_auth_path", codex_home / "auth.json")
    return hermes_path, codex_path


def _read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    """Read one local auth file, returning ``None`` when it is absent.

    Error messages identify the local file and failure class only.  They do
    not include file contents, token values, or raw JSON error snippets.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CodexAuthError(
            f"Codex auth file {str(path)!r} could not be read as JSON "
            f"({type(exc).__name__})"
        ) from exc
    if not isinstance(data, dict):
        raise CodexAuthError(f"Codex auth file {str(path)!r} must contain a JSON object")
    return data


def _nonempty_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _hermes_access_token(data: Mapping[str, Any]) -> Optional[str]:
    """Extract a token from Hermes's provider or credential-pool shape."""
    providers = data.get("providers")
    if isinstance(providers, Mapping):
        provider = providers.get("openai-codex")
        if isinstance(provider, Mapping):
            tokens = provider.get("tokens")
            if isinstance(tokens, Mapping):
                token = _nonempty_string(tokens.get("access_token"))
                if token:
                    return token

    pool = data.get("credential_pool")
    entries = pool.get("openai-codex") if isinstance(pool, Mapping) else None
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            status = str(entry.get("last_status") or "").strip().lower()
            if status in {"dead", "exhausted"}:
                continue
            token = _nonempty_string(
                entry.get("access_token") or entry.get("runtime_api_key")
            )
            if token:
                return token
    return None


def _codex_cli_access_token(data: Mapping[str, Any]) -> Optional[str]:
    tokens = data.get("tokens")
    if not isinstance(tokens, Mapping):
        return None
    return _nonempty_string(tokens.get("access_token"))


def resolve_codex_credentials(param: Mapping[str, Any]) -> CodexCredentials:
    """Resolve an access token from Hermes/Codex's supported local stores.

    ``auth_source`` accepts:

    * ``hermes`` (default): ``$HERMES_HOME/auth.json`` or ``~/.hermes/auth.json``;
    * ``codex_cli``: ``$CODEX_HOME/auth.json`` or ``~/.codex/auth.json``;
    * ``auto``: Hermes first, then the Codex CLI store.

    This is deliberately narrower than accepting arbitrary token values from
    YAML or undocumented environment variables.  Re-authentication and token
    refresh remain the responsibility of Hermes/Codex's own login flows.
    """
    source = str(param.get("auth_source", "hermes")).strip().lower()
    if source not in {"hermes", "codex_cli", "auto"}:
        raise ValueError(
            "model.auth_source must be one of 'hermes', 'codex_cli', or 'auto'"
        )

    hermes_path, codex_path = _supported_auth_paths(param)
    candidates: List[Tuple[str, Path, Any]] = []
    if source in {"hermes", "auto"}:
        candidates.append(("hermes", hermes_path, _hermes_access_token))
    if source in {"codex_cli", "auto"}:
        candidates.append(("codex_cli", codex_path, _codex_cli_access_token))

    for source_name, path, extractor in candidates:
        data = _read_json_object(path)
        if data is None:
            continue
        token = extractor(data)
        if token:
            return CodexCredentials(
                access_token=token,
                source=f"{source_name}:{path}",
                base_url=str(param.get("base_url", DEFAULT_CODEX_BASE_URL)).rstrip("/"),
            )

    hint = (
        "Run `hermes auth add openai-codex` or log in with the Codex CLI"
        if source == "auto"
        else (
            "Run `hermes auth add openai-codex`"
            if source == "hermes"
            else "Log in with the Codex CLI"
        )
    )
    raise CodexAuthError(
        f"No usable OpenAI Codex OAuth token found via auth_source={source!r}. {hint}."
    )


def _chatgpt_account_id(access_token: str) -> Optional[str]:
    """Best-effort account-id extraction used by Hermes's Codex headers."""
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        auth_claim = claims.get("https://api.openai.com/auth", {})
        return (
            _nonempty_string(auth_claim.get("chatgpt_account_id"))
            if isinstance(auth_claim, Mapping)
            else None
        )
    except (ValueError, TypeError, UnicodeError, binascii.Error, json.JSONDecodeError):
        return None


def codex_default_headers(access_token: str, base_url: str) -> Dict[str, str]:
    """Build the first-party-shaped headers used by Hermes's direct transport."""
    headers: Dict[str, str] = {}
    if "chatgpt.com" in base_url.lower():
        headers.update(
            {
                "User-Agent": "codex_cli_rs/0.0.0 (LibSignal)",
                "originator": "codex_cli_rs",
            }
        )
        account_id = _chatgpt_account_id(access_token)
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
    return headers


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    try:
        return getattr(value, name, default)
    except Exception:
        # Some OpenAI SDK convenience properties (notably Response.output_text)
        # iterate ``response.output`` and can raise when the Codex terminal
        # event carries output=null.  A missing/invalid optional field should
        # let the stream parser continue to event-derived content.
        return default


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            return text
        return ""
    text = getattr(value, "text", None)
    return text if isinstance(text, str) else ""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return _text_value(content)
    return "".join(_text_value(part) for part in content)


def messages_to_responses_input(
    messages: Sequence[Mapping[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert LibSignal chat messages to the Codex Responses input shape."""
    instructions: List[str] = []
    items: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = _content_text(message.get("content", ""))
        if role in {"system", "developer"}:
            if content:
                instructions.append(content)
            continue
        if role not in {"user", "assistant"} or not content:
            continue
        text_type = "input_text" if role == "user" else "output_text"
        items.append(
            {
                "role": role,
                "content": [{"type": text_type, "text": content}],
            }
        )
    return "\n\n".join(instructions), items


def _output_item_text(item: Any) -> str:
    item_type = str(_field(item, "type", "") or "").strip().lower()
    if item_type == "reasoning":
        return ""
    if item_type in {"message", "output_message"}:
        return _content_text(_field(item, "content", ""))
    if item_type in {"output_text", "text"}:
        return _text_value(item)
    return ""


def _safe_error_code(error: Any) -> str:
    code = _field(error, "code")
    if code is None:
        code = _field(error, "type")
    return str(code).strip() if code is not None else "unknown"


def _choices_text(response: Any) -> str:
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)):
        return ""
    parts: List[str] = []
    for choice in choices:
        message = _field(choice, "message")
        content = _field(message, "content") if message is not None else None
        text = _content_text(content)
        if text:
            parts.append(text)
    return "\n".join(parts)


def extract_codex_response_text(response: Any) -> str:
    """Extract visible answer text from SDK objects or raw JSON-like shapes.

    The Codex backend has returned both populated ``output`` arrays and
    responses where only ``output_text`` is usable.  It can also expose
    reasoning/function-call items alongside the final message.  This helper
    prefers visible output and never treats reasoning-only content as a
    traffic action.
    """
    if response is None:
        raise RuntimeError("OpenAI Codex returned an empty response")

    error = _field(response, "error")
    if error:
        raise RuntimeError(
            "OpenAI Codex returned an error "
            f"(code={_safe_error_code(error)})"
        )

    output_text = _field(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = _field(response, "output")
    if isinstance(output, (list, tuple)):
        parts = [text for text in (_output_item_text(item) for item in output) if text]
        if parts:
            return "\n".join(parts)

    choices_text = _choices_text(response)
    if choices_text.strip():
        return choices_text

    raise RuntimeError("OpenAI Codex returned no visible output text")


def _response_event_type(event: Any) -> str:
    value = _field(event, "type", "")
    return value.strip().lower() if isinstance(value, str) else ""


def _response_item_id(item: Any) -> Optional[str]:
    value = _field(item, "id")
    return value if isinstance(value, str) and value else None


def _response_item_phase(item: Any) -> Optional[str]:
    value = _field(item, "phase")
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _is_non_visible_phase(phase: Optional[str]) -> bool:
    # Codex can stream commentary/analysis message items through the same
    # output_text.delta event used for the final answer.  Those are useful for
    # a UI but must not become the traffic controller's visible completion.
    return phase in {"commentary", "analysis"}


def _visible_output_item_text(item: Any) -> str:
    """Extract final-answer text from one Responses output item."""
    if _is_non_visible_phase(_response_item_phase(item)):
        return ""

    item_type = str(_field(item, "type", "") or "").strip().lower()
    if item_type in {"output_text", "text"}:
        return _text_value(item)
    if item_type not in {"message", "output_message"}:
        return ""

    content = _field(item, "content", [])
    if not isinstance(content, (list, tuple)):
        return _text_value(content)
    parts: List[str] = []
    for part in content:
        part_type = str(_field(part, "type", "") or "").strip().lower()
        if part_type in {"output_text", "text"}:
            text = _text_value(part)
            if text:
                parts.append(text)
    return "".join(parts)


def extract_codex_stream_text(event_stream: Any) -> str:
    """Consume a streamed Codex Responses result and return visible text.

    The ChatGPT Codex endpoint requires ``stream=true``.  Its terminal
    ``response.completed`` event may contain a null, empty, or otherwise
    changing ``response.output`` field, so text is reconstructed from stream
    events instead of relying on the SDK to build a complete Response object.
    ``response.output_text.delta`` is preferred, with completed output items
    and terminal fields used as compatibility fallbacks.

    A non-iterable complete response is accepted as a small compatibility
    escape hatch for SDK fakes and older callers; the live backend always asks
    the endpoint for a stream.
    """
    if event_stream is None:
        raise RuntimeError("OpenAI Codex returned an empty response stream")

    # Pydantic Response objects and the simple objects commonly used by tests
    # expose response fields but are not event iterables.  Do not iterate a
    # mapping response's keys as though each key were an SSE event.
    if (
        not isinstance(event_stream, (list, tuple))
        and _field(event_stream, "type") is None
        and any(
            _field(event_stream, name) is not None
            for name in ("output_text", "output", "choices", "error")
        )
    ):
        return extract_codex_response_text(event_stream)

    try:
        events = iter(event_stream)
    except TypeError:
        return extract_codex_response_text(event_stream)

    # Keep the item id with each delta so a late output_item.done event can
    # tell us that earlier deltas belonged to commentary rather than the final
    # answer.  The phase snapshot covers providers that omit item ids.
    delta_records: List[Tuple[Optional[str], Optional[str], str]] = []
    done_text_records: List[Tuple[Optional[str], Optional[str], str]] = []
    output_items: List[Any] = []
    phases: Dict[str, Optional[str]] = {}
    active_item_id: Optional[str] = None
    active_phase: Optional[str] = None
    terminal_response: Any = None
    saw_terminal = False

    for event in events:
        event_type = _response_event_type(event)
        if event_type == "error":
            error = _field(event, "error", event)
            code = _field(error, "code") or _field(event, "code") or "unknown"
            message = _field(error, "message") or _field(event, "message")
            detail = str(message).strip() if message is not None else "stream emitted error event"
            raise RuntimeError(
                f"OpenAI Codex stream error (code={str(code).strip()}): {detail}"
            )

        if event_type == "response.output_item.added":
            item = _field(event, "item")
            active_item_id = _response_item_id(item)
            active_phase = _response_item_phase(item)
            if active_item_id is not None:
                phases[active_item_id] = active_phase
            continue

        if event_type == "response.output_text.delta":
            delta = _field(event, "delta", "")
            if not isinstance(delta, str) or not delta:
                continue
            item_id = _field(event, "item_id")
            if not isinstance(item_id, str) or not item_id:
                item_id = active_item_id
            phase = phases.get(item_id, active_phase) if item_id else active_phase
            delta_records.append((item_id, phase, delta))
            continue

        if event_type == "response.output_text.done":
            text = _field(event, "text", "")
            if isinstance(text, str) and text:
                item_id = _field(event, "item_id")
                if not isinstance(item_id, str) or not item_id:
                    item_id = active_item_id
                phase = phases.get(item_id, active_phase) if item_id else active_phase
                done_text_records.append((item_id, phase, text))
            continue

        if event_type == "response.output_item.done":
            item = _field(event, "item")
            if item is not None:
                output_items.append(item)
                item_id = _response_item_id(item)
                if item_id is not None:
                    done_phase = _response_item_phase(item)
                    if done_phase is not None or item_id not in phases:
                        phases[item_id] = done_phase
            continue

        if event_type in {"response.completed", "response.incomplete", "response.failed"}:
            saw_terminal = True
            terminal_response = _field(event, "response")
            if event_type == "response.failed":
                error = _field(terminal_response, "error")
                code = _field(error, "code") or "unknown"
                message = _field(error, "message") or "Codex response failed"
                raise RuntimeError(
                    f"OpenAI Codex stream failed (code={str(code).strip()}): "
                    f"{str(message).strip()}"
                )
            break

    visible_deltas: List[str] = []
    for item_id, phase_snapshot, text in delta_records:
        phase = phases.get(item_id, phase_snapshot) if item_id else phase_snapshot
        if not _is_non_visible_phase(phase):
            visible_deltas.append(text)
    if visible_deltas:
        return "".join(visible_deltas)

    visible_done_text: List[str] = []
    for item_id, phase_snapshot, text in done_text_records:
        phase = phases.get(item_id, phase_snapshot) if item_id else phase_snapshot
        if not _is_non_visible_phase(phase):
            visible_done_text.append(text)
    if visible_done_text:
        return "".join(visible_done_text)

    item_text = [text for item in output_items if (text := _visible_output_item_text(item))]
    if item_text:
        return "\n".join(item_text)

    terminal_text = _field(terminal_response, "output_text")
    if isinstance(terminal_text, str) and terminal_text.strip():
        return terminal_text

    # Some compatible servers omit output_item.done but still provide a
    # conventional output array in the terminal response.  Filter it through
    # the same visible-item rules rather than trusting output_text alone.
    terminal_output = _field(terminal_response, "output")
    if isinstance(terminal_output, (list, tuple)):
        item_text = [
            text
            for item in terminal_output
            if (text := _visible_output_item_text(item))
        ]
        if item_text:
            return "\n".join(item_text)

    if not saw_terminal:
        raise RuntimeError("OpenAI Codex stream did not emit a terminal response")
    raise RuntimeError("OpenAI Codex stream returned no visible output text")


def build_codex_request(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str,
    reasoning_enabled: bool = True,
    include_reasoning: bool = False,
    extra_body: Optional[Mapping[str, Any]] = None,
    request_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a minimal Hermes-compatible direct Codex Responses request."""
    instructions, input_items = messages_to_responses_input(messages)
    request: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "store": False,
    }
    if reasoning_enabled:
        request["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        if include_reasoning:
            request["include"] = ["reasoning.encrypted_content"]
    else:
        request["include"] = []
    # Hermes's direct ``openai-codex`` transport omits both token-limit fields
    # for the ChatGPT Codex backend.  Keep ``max_output_tokens`` as a local
    # compatibility/configuration value, but never put either spelling on the
    # wire.
    body_overrides = dict(extra_body or {})
    body_overrides.pop("max_output_tokens", None)
    body_overrides.pop("max_tokens", None)
    request["extra_body"] = body_overrides
    if request_overrides:
        request.update(dict(request_overrides))

    # Keep generic overrides from reintroducing either unsupported token field.
    request.pop("max_output_tokens", None)
    request.pop("max_tokens", None)
    request_body = dict(request.get("extra_body") or {})
    request_body.pop("max_output_tokens", None)
    request_body.pop("max_tokens", None)
    request["extra_body"] = request_body
    # The ChatGPT Codex backend rejects non-streaming Responses requests.
    # Keep this invariant even if a generic override mapping contains a stale
    # ``stream`` value.
    request["stream"] = True
    return request


def prepare_messages(
    messages: Sequence[Mapping[str, Any]],
    suffix: str = DEFAULT_API_FORMAT_SUFFIX,
) -> List[Dict[str, Any]]:
    """Copy messages and append the strict boxed-answer instruction."""
    prepared = [dict(message) for message in messages]
    for message in reversed(prepared):
        if message.get("role") == "user":
            message["content"] = str(message.get("content", "")) + suffix
            break
    return prepared
