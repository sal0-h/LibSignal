from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.llm_tsc_backend import CodexResponsesBackend, create_backend
from agent.llm_tsc_codex import (
    CodexAuthError,
    CodexCredentials,
    build_codex_request,
    extract_codex_response_text,
    extract_codex_stream_text,
    prepare_messages,
    resolve_codex_credentials,
)


class _FakeResponses:
    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        user_text = request["input"][0]["content"][0]["text"]
        signal = user_text.split(" Keep the think block", 1)[0].split()[-1]
        text = "\\boxed{" + signal + "}"
        item = SimpleNamespace(
            type="message",
            id=f"msg_{signal}",
            phase="final_answer",
            content=[SimpleNamespace(type="output_text", text=text)],
        )
        return [
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    type="message", id=f"msg_{signal}", phase="final_answer"
                ),
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                item_id=f"msg_{signal}",
                delta=text,
            ),
            SimpleNamespace(type="response.output_item.done", item=item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", output=None),
            ),
        ]


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


class CodexAuthTests(unittest.TestCase):
    def test_resolves_hermes_store_without_exposing_token_in_repr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "auth.json"
            path.write_text(
                json.dumps(
                    {
                        "providers": {
                            "openai-codex": {
                                "tokens": {"access_token": "test-token-value"}
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            credentials = resolve_codex_credentials(
                {"auth_source": "hermes", "auth_path": str(path)}
            )

        self.assertEqual(credentials.access_token, "test-token-value")
        self.assertNotIn("test-token-value", repr(credentials))

    def test_auth_source_can_select_codex_cli_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "auth.json"
            path.write_text(
                json.dumps({"tokens": {"access_token": "cli-token"}}),
                encoding="utf-8",
            )
            credentials = resolve_codex_credentials(
                {"auth_source": "codex_cli", "codex_auth_path": str(path)}
            )
        self.assertEqual(credentials.source.split(":", 1)[0], "codex_cli")

    def test_missing_supported_auth_is_actionable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(CodexAuthError, "auth_source='hermes'"):
                resolve_codex_credentials(
                    {
                        "auth_source": "hermes",
                        "auth_path": str(Path(temp_dir) / "missing.json"),
                    }
                )


class CodexWireTests(unittest.TestCase):
    def test_request_uses_responses_shape_and_strict_suffix(self):
        messages = prepare_messages(
            [
                {"role": "system", "content": "Traffic system"},
                {"role": "user", "content": "Choose ETWT"},
            ]
        )
        request = build_codex_request(
            messages,
            model="gpt-5.6-luna",
            max_output_tokens=128,
            reasoning_effort="low",
        )

        self.assertEqual(request["model"], "gpt-5.6-luna")
        self.assertEqual(request["instructions"], "Traffic system")
        self.assertEqual(request["input"][0]["role"], "user")
        self.assertIn("\\boxed{ETWT}", request["input"][0]["content"][0]["text"])
        self.assertFalse(request["store"])
        self.assertNotIn("max_tokens", request)
        self.assertNotIn("max_output_tokens", request)
        self.assertNotIn("max_tokens", request["extra_body"])
        self.assertNotIn("max_output_tokens", request["extra_body"])
        self.assertTrue(request["stream"])
        self.assertEqual(request["reasoning"]["effort"], "low")
        self.assertNotIn("temperature", request)

    def test_response_extraction_handles_output_text_and_output_items(self):
        self.assertEqual(
            extract_codex_response_text(SimpleNamespace(output_text="\\boxed{ETWT}")),
            "\\boxed{ETWT}",
        )
        response = {
            "output": [
                {"type": "reasoning", "summary": [{"text": "ignored"}]},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "\\boxed{NLSL}"}],
                },
            ]
        }
        self.assertEqual(extract_codex_response_text(response), "\\boxed{NLSL}")

    def test_response_extraction_supports_chat_compatibility_fallback(self):
        response = {
            "choices": [{"message": {"content": "\\boxed{ELWL}"}}]
        }
        self.assertEqual(extract_codex_response_text(response), "\\boxed{ELWL}")

    def test_stream_parser_uses_visible_deltas_when_terminal_output_is_null(self):
        events = [
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "thinking", "phase": "commentary"},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "thinking",
                "delta": "not the answer",
            },
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "answer", "phase": "final_answer"},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "answer",
                "delta": r"\boxed{ET",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "answer",
                "delta": r"WT}",
            },
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": None},
            },
        ]

        self.assertEqual(extract_codex_stream_text(events), r"\boxed{ETWT}")

    def test_stream_parser_reconstructs_from_completed_output_item(self):
        events = [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "ignored"}],
                },
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": r"\boxed{NLSL}"}],
                },
            },
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": None},
            },
        ]

        self.assertEqual(extract_codex_stream_text(events), r"\boxed{NLSL}")

    def test_backend_batches_in_request_order(self):
        client = _FakeClient()
        backend = CodexResponsesBackend(
            {
                "model_id": "gpt-5.6-luna",
                "parallelism": 2,
            },
            credentials=CodexCredentials("test-token", "test"),
            client=client,
        )
        results = backend.complete_many(
            [
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "ETWT"},
                ],
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "NLSL"},
                ],
            ]
        )
        self.assertEqual(results, ["\\boxed{ETWT}", "\\boxed{NLSL}"])
        self.assertEqual(len(client.responses.requests), 2)
        self.assertTrue(
            all(request["stream"] is True for request in client.responses.requests)
        )
        self.assertTrue(
            all(
                "max_tokens" not in request
                for request in client.responses.requests
            )
        )
        self.assertTrue(
            all(
                "max_output_tokens" not in request
                and "max_tokens" not in request.get("extra_body", {})
                and "max_output_tokens" not in request.get("extra_body", {})
                for request in client.responses.requests
            )
        )

    def test_factory_selects_codex_aliases(self):
        # Credentials are resolved before the client is created, so a missing
        # auth file is the expected boundary for this configuration-only test.
        with tempfile.TemporaryDirectory() as temp_dir:
            params = {
                "backend": "codex",
                "auth_source": "hermes",
                "auth_path": str(Path(temp_dir) / "missing.json"),
            }
            with self.assertRaises(CodexAuthError):
                create_backend(params)


if __name__ == "__main__":
    unittest.main()
