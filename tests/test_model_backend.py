"""Backend protocol tests use mocks only: no network or model inference."""

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from pde_rsi.model_backend import ModelBackend, ModelCallError, ModelConfig, _NoRedirect, _request_body
from pde_rsi.storage import PROJECT_ROOT


class ModelBackendTests(unittest.TestCase):
    def setUp(self):
        test_root = PROJECT_ROOT / ".runtime/tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="model-backend-tests-", dir=test_root)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "call"
        self.env = patch.dict(os.environ, {"PDE_RSI_API_KEY": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.messages = [{"role": "user", "content": "Return a JSON proposal"}]
        self.schema = {"type": "object", "properties": {"n": {"type": "integer"}},
                       "required": ["n"], "additionalProperties": False}

    def backend(self, **options):
        return ModelBackend(ModelConfig("http://localhost:8000/v1", "selected-model", **options))

    def response(self, content='{"n": 65}', **overrides):
        result = {"model": "selected-model-version", "choices": [
            {"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}
        result.update(overrides)
        return io.BytesIO(json.dumps(result).encode())

    def call(self, backend=None):
        return (backend or self.backend()).complete(self.messages, self.schema, self.directory, timeout=3.0)

    @patch("pde_rsi.model_backend.urlopen")
    def test_success_records_requested_and_resolved_models(self, mocked):
        mocked.return_value = self.response()
        result = self.call()
        self.assertEqual(result["proposal"], {"n": 65})
        self.assertEqual(result["resolved_model"], "selected-model-version")
        self.assertEqual(result["usage"], {"prompt_tokens": 12, "completion_tokens": 8,
                                           "total_tokens": 20, "available": True})
        self.assertGreaterEqual(result["wall_seconds"], 0)
        request = mocked.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(body["model"], "selected-model")
        self.assertEqual(body["messages"], self.messages)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_tokens"], 2000)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(mocked.call_args.kwargs["timeout"], 3.0)
        self.assertEqual(request.get_method(), "POST")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(json.loads((self.directory / "request.json").read_text()), body)
        self.assertEqual(json.loads((self.directory / "response.json").read_text())["model"], "selected-model-version")
        mocked.assert_called_once()

    @patch("pde_rsi.model_backend.urlopen")
    def test_complete_url_not_appended_twice(self, mocked):
        mocked.return_value = self.response()
        backend = ModelBackend(ModelConfig("http://127.0.0.1:8000/v1/chat/completions/", "selected-model"))
        self.call(backend)
        self.assertEqual(mocked.call_args.args[0].full_url, "http://127.0.0.1:8000/v1/chat/completions")

    @patch("pde_rsi.model_backend.urlopen")
    def test_json_schema_format(self, mocked):
        mocked.return_value = self.response()
        self.call(self.backend(response_format="json_schema"))
        body = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(body["response_format"], {"type": "json_schema", "json_schema": {
            "name": "pde_rsi_proposal", "strict": True, "schema": self.schema}})

    @patch("pde_rsi.model_backend.urlopen")
    def test_none_format_omits_response_format(self, mocked):
        mocked.return_value = self.response()
        self.call(self.backend(response_format="none"))
        self.assertNotIn("response_format", json.loads(mocked.call_args.args[0].data))

    @patch("pde_rsi.model_backend.urlopen")
    def test_missing_hosted_key_does_not_call_model(self, mocked):
        backend = ModelBackend(ModelConfig("https://endpoint.example/v1", "selected-model"))
        with self.assertRaisesRegex(ModelCallError, "missing_api_key") as caught:
            self.call(backend)
        mocked.assert_not_called()
        self.assertFalse(caught.exception.call_attempted)
        self.assertFalse(caught.exception.usage["available"])
        self.assertTrue((self.directory / "request.json").exists())
        self.assertEqual(json.loads((self.directory / "response.json").read_text())["error"]["type"], "missing_api_key")

    @patch("pde_rsi.model_backend.urlopen")
    def test_credentials_only_in_transport_header(self, mocked):
        os.environ["PDE_RSI_API_KEY"] = "secret-for-test-redaction"
        mocked.return_value = self.response(extra="Provider echoed secret-for-test-redaction")
        backend = ModelBackend(ModelConfig("https://endpoint.example/v1", "selected-model"))
        self.messages[0]["content"] += " secret-for-test-redaction"
        self.call(backend)
        self.assertEqual(mocked.call_args.args[0].get_header("Authorization"), "Bearer secret-for-test-redaction")
        for name in ("request.json", "response.json"):
            text = (self.directory / name).read_text()
            self.assertNotIn("secret-for-test-redaction", text)
            self.assertNotIn("Authorization", text)
        self.assertNotIn("secret-for-test-redaction", json.dumps(backend.config.to_dict()))

    @patch("pde_rsi.model_backend.urlopen")
    def test_http_failure_is_sanitized_and_not_retried(self, mocked):
        os.environ["PDE_RSI_API_KEY"] = "test-secret"
        mocked.side_effect = HTTPError("http://localhost", 429, "Authorization: Bearer test-secret", {}, None)
        with self.assertRaises(ModelCallError) as caught:
            self.call()
        self.assertIn("HTTP 429", str(caught.exception))
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertTrue(caught.exception.call_attempted)
        self.assertIsNone(caught.exception.usage["total_tokens"])
        self.assertNotIn("test-secret", (self.directory / "response.json").read_text())
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)
        mocked.assert_called_once()

    @patch("pde_rsi.model_backend.urlopen")
    def test_network_failure_is_sanitized_and_not_retried(self, mocked):
        mocked.side_effect = URLError("Authorization: Bearer must-not-log")
        with self.assertRaisesRegex(ModelCallError, "transport_error") as caught:
            self.call()
        self.assertNotIn("must-not-log", str(caught.exception))
        self.assertTrue(caught.exception.call_attempted)
        mocked.assert_called_once()

    @patch("pde_rsi.model_backend.urlopen")
    def test_malformed_outer_response(self, mocked):
        mocked.return_value = io.BytesIO(b"not JSON: Bearer secret")
        with self.assertRaisesRegex(ModelCallError, "malformed_response_json"):
            self.call()
        self.assertNotIn("Bearer secret", (self.directory / "response.json").read_text())
        mocked.assert_called_once()

    @patch("pde_rsi.model_backend.urlopen")
    def test_malformed_content_preserves_known_usage(self, mocked):
        mocked.return_value = self.response("not a proposal")
        with self.assertRaisesRegex(ModelCallError, "malformed_content") as caught:
            self.call()
        self.assertEqual(caught.exception.usage["total_tokens"], 20)
        self.assertEqual(caught.exception.resolved_model, "selected-model-version")

    @patch("pde_rsi.model_backend.urlopen")
    def test_missing_usage_is_unknown_not_zero(self, mocked):
        mocked.return_value = self.response(usage=None, model=None)
        result = self.call()
        self.assertEqual(result["usage"], {"prompt_tokens": None, "completion_tokens": None,
                                          "total_tokens": None, "available": False})
        self.assertIsNone(result["resolved_model"])

    @patch("pde_rsi.model_backend.urlopen")
    def test_partial_usage_does_not_infer_total(self, mocked):
        mocked.return_value = self.response(usage={"prompt_tokens": 12, "completion_tokens": 8})
        result = self.call()
        self.assertTrue(result["usage"]["available"])
        self.assertIsNone(result["usage"]["total_tokens"])

    @patch("pde_rsi.model_backend.urlopen")
    def test_invalid_usage_values_remain_unknown(self, mocked):
        mocked.return_value = self.response(usage={"prompt_tokens": True, "completion_tokens": -1, "total_tokens": "20"})
        self.assertFalse(self.call()["usage"]["available"])

    @patch("pde_rsi.model_backend.urlopen")
    def test_optional_fenced_json(self, mocked):
        mocked.return_value = self.response('```json\n{"n": 129}\n```')
        self.assertEqual(self.call()["proposal"], {"n": 129})

    @patch("pde_rsi.model_backend.urlopen")
    def test_list_content_text(self, mocked):
        mocked.return_value = self.response([{"type": "text", "text": '{"n": 33}'}])
        self.assertEqual(self.call()["proposal"], {"n": 33})

    @patch("pde_rsi.model_backend.urlopen")
    def test_non_object_and_nonfinite_proposals_rejected(self, mocked):
        for index, content in enumerate(("[1, 2]", '{"value": NaN}')):
            self.directory = Path(self.temporary.name) / str(index)
            mocked.return_value = self.response(content)
            with self.assertRaises(ModelCallError):
                self.call()

    @patch("pde_rsi.model_backend.urlopen")
    def test_incomplete_response_rejected_even_if_parseable(self, mocked):
        mocked.return_value = self.response(choices=[{"message": {"content": '{"n": 65}'}, "finish_reason": "length"}])
        with self.assertRaisesRegex(ModelCallError, "incomplete_response"):
            self.call()

    @patch("pde_rsi.model_backend.urlopen")
    def test_existing_audit_files_are_preserved(self, mocked):
        mocked.return_value = self.response()
        self.call()
        before = (self.directory / "response.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.call()
        self.assertEqual(before, (self.directory / "response.json").read_bytes())
        mocked.assert_called_once()

    @patch("pde_rsi.model_backend.urlopen")
    def test_outside_project_is_rejected_without_call(self, mocked):
        with self.assertRaisesRegex(ValueError, "inside"):
            self.backend().complete(self.messages, self.schema, Path("/home/lu895/forbidden-output"), 3)
        mocked.assert_not_called()

    def test_embedded_credentials_and_invalid_urls_rejected(self):
        for url in ("ftp://localhost/v1", "file:///tmp/model", "https://user:secret@example.org/v1",
                    "https://example.org/v1?api_key=secret", "https://example.org/v1#secret", "https://example.org:bad/v1"):
            with self.subTest(url=url), self.assertRaises(ValueError) as caught:
                ModelConfig(url, "model")
            self.assertNotIn("secret", str(caught.exception))

    def test_no_implicit_redirects(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://other.example/"))

    @patch("pde_rsi.model_backend.urlopen")
    def test_multiline_key_rejected_without_call(self, mocked):
        os.environ["PDE_RSI_API_KEY"] = "secret-line\nanother-secret"
        with self.assertRaisesRegex(ModelCallError, "invalid_api_key") as caught:
            self.call()
        self.assertNotIn("secret-line", str(caught.exception))
        mocked.assert_not_called()

    @patch("pde_rsi.model_backend.urlopen")
    def test_invalid_timeout_fails_before_network(self, mocked):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.backend().complete(self.messages, self.schema, self.directory, timeout)
        mocked.assert_not_called()


    def bedrock(self, **options):
        return ModelBackend(ModelConfig(
            "https://bedrock-runtime.us-east-1.amazonaws.com", "us.anthropic.test-model-v1:0",
            protocol="bedrock_converse", **options))

    def bedrock_response(self, content='{"n": 65}', **overrides):
        result = {"output": {"message": {"role": "assistant", "content": [{"text": content}]}},
                  "stopReason": "end_turn", "usage": {"inputTokens": 19, "outputTokens": 11, "totalTokens": 30}}
        result.update(overrides)
        return io.BytesIO(json.dumps(result).encode())

    @patch("pde_rsi.model_backend.urlopen")
    def test_bedrock_payload_usage_and_pinned_route_audit(self, mocked):
        os.environ["PDE_RSI_API_KEY"] = "synthetic-bedrock-key"
        mocked.return_value = self.bedrock_response()
        self.messages = [{"role": "system", "content": "Return JSON"},
                         {"role": "user", "content": "Choose n"},
                         {"role": "assistant", "content": '{"n": 33}'},
                         {"role": "user", "content": "Improve"}]
        result = self.call(self.bedrock())
        self.assertEqual(result["proposal"], {"n": 65})
        self.assertIsNone(result["resolved_model"])
        self.assertEqual(result["usage"], {"prompt_tokens": 19, "completion_tokens": 11,
                                           "total_tokens": 30, "available": True})
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url,
                         "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.test-model-v1%3A0/converse")
        self.assertEqual(request.get_header("Authorization"), "Bearer synthetic-bedrock-key")
        body = json.loads(request.data)
        self.assertEqual(body["system"], [{"text": "Return JSON"}])
        self.assertEqual(body["messages"], [
            {"role": "user", "content": [{"text": "Choose n"}]},
            {"role": "assistant", "content": [{"text": '{"n": 33}'}]},
            {"role": "user", "content": [{"text": "Improve"}]}])
        self.assertEqual(body["inferenceConfig"], {"maxTokens": 2000, "temperature": 0.0})
        self.assertNotIn("model", body)
        self.assertNotIn("response_format", body)
        audit = json.loads((self.directory / "request.json").read_text())
        self.assertEqual(audit["body"], body)
        self.assertEqual(audit["requested_model"], "us.anthropic.test-model-v1:0")
        self.assertEqual(audit["url"], request.full_url)
        self.assertNotIn("synthetic-bedrock-key", (self.directory / "request.json").read_text())
        mocked.assert_called_once()

    @patch("pde_rsi.model_backend.urlopen")
    def test_bedrock_file_key_read_only_at_call_and_redacted(self, mocked):
        key_file = Path(self.temporary.name) / "synthetic-key"
        key_file.write_text("synthetic-private-file-key\n")
        os.environ["PDE_RSI_API_KEY"] = "ignored-environment-key"
        with patch.object(Path, "read_text", side_effect=AssertionError("configuration must not read the file")):
            backend = self.bedrock(api_key_file=str(key_file))
            self.assertEqual(backend.config.to_dict()["api_key_file"], str(key_file))
        mocked.return_value = self.bedrock_response(extra="echo synthetic-private-file-key")
        self.call(backend)
        self.assertEqual(mocked.call_args.args[0].get_header("Authorization"), "Bearer synthetic-private-file-key")
        self.assertNotIn("synthetic-private-file-key", (self.directory / "response.json").read_text())
        self.assertNotIn("synthetic-private-file-key", json.dumps(backend.config.to_dict()))

    @patch("pde_rsi.model_backend.urlopen")
    def test_openai_compatible_file_key(self, mocked):
        key_file = Path(self.temporary.name) / "synthetic-openai-compatible-key"
        key_file.write_text("synthetic-local-file-key")
        mocked.return_value = self.response()
        self.call(self.backend(api_key_file=str(key_file)))
        self.assertEqual(mocked.call_args.args[0].get_header("Authorization"), "Bearer synthetic-local-file-key")

    @patch("pde_rsi.model_backend.urlopen")
    def test_key_file_unavailable_sanitized_before_model_call(self, mocked):
        key_file = Path(self.temporary.name) / "missing-private-key"
        with self.assertRaisesRegex(ModelCallError, "api_key_file_unavailable") as caught:
            self.call(self.bedrock(api_key_file=str(key_file)))
        self.assertFalse(caught.exception.call_attempted)
        self.assertNotIn("missing-private-key", str(caught.exception))
        self.assertEqual(json.loads((self.directory / "response.json").read_text())["error"]["type"], "api_key_file_unavailable")
        mocked.assert_not_called()

    def test_key_file_outside_project_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside"):
            self.backend(api_key_file="/home/lu895/private-key")

    def test_bedrock_rejects_non_aws_and_insecure_origins(self):
        urls = ("http://bedrock-runtime.us-east-1.amazonaws.com",
                "https://localhost:8000", "https://bedrock-runtime.us-east-1.amazonaws.com.evil.example",
                "https://bedrock-runtime.us-east-1.amazonaws.com:8080",
                "https://bedrock-runtime.us-east-1.amazonaws.com/v1")
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                ModelConfig(url, "selected-model", protocol="bedrock_converse")

    @patch("pde_rsi.model_backend.urlopen")
    def test_bedrock_native_schema_format(self, mocked):
        os.environ["PDE_RSI_API_KEY"] = "synthetic-bedrock-key"
        mocked.return_value = self.bedrock_response()
        self.call(self.bedrock(response_format="json_schema"))
        body = json.loads(mocked.call_args.args[0].data)
        text_format = body["outputConfig"]["textFormat"]
        self.assertEqual(text_format["type"], "json_schema")
        definition = text_format["structure"]["jsonSchema"]
        self.assertEqual(definition["name"], "pde_rsi_proposal")
        self.assertEqual(json.loads(definition["schema"]), self.schema)

    @patch("pde_rsi.model_backend.urlopen")
    def test_bedrock_truncation_preserves_actual_usage(self, mocked):
        os.environ["PDE_RSI_API_KEY"] = "synthetic-bedrock-key"
        mocked.return_value = self.bedrock_response(stopReason="max_tokens")
        with self.assertRaisesRegex(ModelCallError, "incomplete_response") as caught:
            self.call(self.bedrock())
        self.assertEqual(caught.exception.usage["total_tokens"], 30)
        self.assertTrue(caught.exception.call_attempted)
        mocked.assert_called_once()

    @patch("pde_rsi.model_backend.urlopen")
    def test_bedrock_reasoning_blocks_do_not_pollute_json(self, mocked):
        os.environ["PDE_RSI_API_KEY"] = "synthetic-bedrock-key"
        mocked.return_value = self.bedrock_response(output={"message": {"role": "assistant", "content": [
            {"reasoningContent": {"reasoningText": {"text": "hidden explanation"}}},
            {"text": "```json\n"}, {"text": '{"n": 129}\n```'}]}})
        self.assertEqual(self.call(self.bedrock())["proposal"], {"n": 129})

    @patch("pde_rsi.model_backend.urlopen")
    def test_bedrock_missing_usage_is_unknown(self, mocked):
        os.environ["PDE_RSI_API_KEY"] = "synthetic-bedrock-key"
        mocked.return_value = self.bedrock_response(usage=None)
        result = self.call(self.bedrock())
        self.assertFalse(result["usage"]["available"])
        self.assertIsNone(result["usage"]["total_tokens"])

    @patch("pde_rsi.model_backend.urlopen")
    def test_bedrock_invalid_role_rejected_before_network(self, mocked):
        self.messages = [{"role": "tool", "content": "unsupported"}]
        with self.assertRaisesRegex(ValueError, "roles"):
            self.call(self.bedrock())
        mocked.assert_not_called()

    @patch("pde_rsi.model_backend.urlopen")
    def test_bedrock_http_failure_sanitized_and_not_retried(self, mocked):
        os.environ["PDE_RSI_API_KEY"] = "synthetic-bedrock-key"
        mocked.side_effect = HTTPError("https://bedrock-runtime.us-east-1.amazonaws.com", 403,
                                       "Bearer synthetic-bedrock-key", {}, None)
        with self.assertRaises(ModelCallError) as caught:
            self.call(self.bedrock())
        self.assertIn("HTTP 403", str(caught.exception))
        self.assertNotIn("synthetic-bedrock-key", str(caught.exception))
        self.assertNotIn("synthetic-bedrock-key", (self.directory / "response.json").read_text())
        mocked.assert_called_once()


    @patch("pde_rsi.model_backend.urlopen")
    def test_prose_with_one_fenced_json_accepted_without_retry(self, mocked):
        mocked.return_value = self.response(
            'I recommend refining the fractional time grid.\n\n```json\n{"n": 129}\n```\n'
            'This should resolve the initial-time error.')
        self.assertEqual(self.call()["proposal"], {"n": 129})
        mocked.assert_called_once()

    @patch("pde_rsi.model_backend.urlopen")
    def test_bedrock_prose_with_fenced_json_and_reasoning(self, mocked):
        os.environ["PDE_RSI_API_KEY"] = "synthetic-bedrock-key"
        mocked.return_value = self.bedrock_response(output={"message": {"role": "assistant", "content": [
            {"reasoningContent": {"reasoningText": {"text": "private explanation"}}},
            {"text": 'Recommended proposal follows.\n```json\n{"n": 65}\n```\nUse this setting.'}]}})
        self.assertEqual(self.call(self.bedrock())["proposal"], {"n": 65})
        mocked.assert_called_once()

    @patch("pde_rsi.model_backend.urlopen")
    def test_multiple_fenced_blocks_rejected_even_if_equal(self, mocked):
        mocked.return_value = self.response('```json\n{"n": 65}\n```\nAlternative:\n```json\n{"n": 65}\n```')
        with self.assertRaisesRegex(ModelCallError, "ambiguous_content") as caught:
            self.call()
        self.assertEqual(caught.exception.usage["total_tokens"], 20)
        mocked.assert_called_once()

    @patch("pde_rsi.model_backend.urlopen")
    def test_conflicting_raw_object_outside_fence_rejected(self, mocked):
        examples = ('Raw proposal: {"n": 33}\n```json\n{"n": 65}\n```',
                    '```json\n{"n": 65}\n```\nBut use {"n": 129}',
                    '{"n": true}\n```json\n{"n": 1}\n```')
        for index, content in enumerate(examples):
            self.directory = Path(self.temporary.name) / f"ambiguous-{index}"
            mocked.return_value = self.response(content)
            with self.assertRaisesRegex(ModelCallError, "ambiguous_content"):
                self.call()
        self.assertEqual(mocked.call_count, len(examples))

    @patch("pde_rsi.model_backend.urlopen")
    def test_identical_raw_object_outside_fence_is_unambiguous(self, mocked):
        mocked.return_value = self.response('Summary {"nested": {"n": 65}}\n```json\n{"nested": {"n": 65}}\n```')
        self.assertEqual(self.call()["proposal"], {"nested": {"n": 65}})

    @patch("pde_rsi.model_backend.urlopen")
    def test_plain_json_with_literal_backticks_stays_valid(self, mocked):
        mocked.return_value = self.response('{"lesson": "Avoid ``` notation in code"}')
        self.assertEqual(self.call()["proposal"], {"lesson": "Avoid ``` notation in code"})

    @patch("pde_rsi.model_backend.urlopen")
    def test_fenced_nonfinite_nonobject_and_duplicate_keys_rejected(self, mocked):
        for index, value in enumerate(('{"value": NaN}', '{"value": 1e999}', '[1,2]', '{"n": 33, "n": 65}')):
            self.directory = Path(self.temporary.name) / f"invalid-fence-{index}"
            mocked.return_value = self.response('Recommended:\n```json\n' + value + '\n```\nUse this.')
            with self.assertRaises(ModelCallError):
                self.call()

    @patch("pde_rsi.model_backend.urlopen")
    def test_nonfinite_outer_json_rejected_safely(self, mocked):
        mocked.return_value = io.BytesIO(b'{"bad": 1e999}')
        with self.assertRaisesRegex(ModelCallError, "malformed_response_json"):
            self.call()
        self.assertEqual(json.loads((self.directory / "response.json").read_text())["error"]["type"],
                         "malformed_response_json")



class TemperatureOmissionTests(unittest.TestCase):
    def test_null_temperature_is_omitted_for_both_protocols(self):
        messages = [{"role": "user", "content": "Return JSON"}]
        bedrock = ModelConfig("https://bedrock-runtime.us-east-1.amazonaws.com", "us.openai.gpt-5.5",
                              protocol="bedrock_converse", response_format="none", temperature=None)
        self.assertEqual(_request_body(bedrock, messages, None)["inferenceConfig"], {"maxTokens": 2000})
        openai = ModelConfig("http://localhost:9999/v1", "mock", temperature=None)
        self.assertNotIn("temperature", _request_body(openai, messages, None))
        with self.assertRaises(ValueError):
            ModelConfig("http://localhost:9999/v1", "mock", temperature=3.0)

    def test_bedrock_cache_tokens_count_as_input(self):
        from pde_rsi.model_backend import _usage
        usage = _usage({"usage": {"inputTokens": 2, "outputTokens": 10, "totalTokens": 40524,
                                  "cacheReadInputTokens": 0, "cacheWriteInputTokens": 38862}}, "bedrock_converse")
        self.assertEqual(usage["prompt_tokens"], 38864)
        self.assertEqual(usage["cache_tokens"], {"cacheWriteInputTokens": 38862})
        plain = _usage({"usage": {"inputTokens": 19, "outputTokens": 11, "totalTokens": 30}}, "bedrock_converse")
        self.assertNotIn("cache_tokens", plain)

if __name__ == "__main__":
    unittest.main()
