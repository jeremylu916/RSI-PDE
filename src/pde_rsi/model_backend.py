"""One-call HTTP model backend with project-local, credential-free audit files.

The backend never retries. Callers can therefore account for every attempted
request, including failures and responses with unavailable token usage.
"""

from dataclasses import asdict, dataclass
from http.client import HTTPException
import ipaddress
import json
import math
import os
import re
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .storage import project_path


_UNKNOWN_USAGE = {"prompt_tokens": None, "completion_tokens": None,
                  "total_tokens": None, "available": False}
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        # A redirect would create another request and might forward credentials.
        return None


def urlopen(request, timeout):
    """One HTTP request, including no implicit HTTP redirect requests."""
    return build_opener(_NoRedirect).open(request, timeout=timeout)


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    api_key_env: str = "PDE_RSI_API_KEY"
    max_output_tokens: int = 2000
    temperature: float | None = 0.0  # None omits the field for models that reject it
    response_format: str = "json_object"
    protocol: str = "openai_compatible"
    api_key_file: str | None = None

    def __post_init__(self):
        if self.protocol not in {"openai_compatible", "bedrock_converse"}:
            raise ValueError("protocol must be openai_compatible or bedrock_converse")
        _endpoint_url(self)
        if self.api_key_file is not None:
            if not isinstance(self.api_key_file, str):
                raise ValueError("api_key_file must be a project-local path string")
            project_path(self.api_key_file)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("A model identifier is required")
        if not isinstance(self.api_key_env, str) or not self.api_key_env.strip():
            raise ValueError("An API-key environment-variable name is required")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        if self.temperature is not None and (
                not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature)
                or not 0 <= self.temperature <= 2):
            raise ValueError("temperature must be null or finite and between 0 and 2")
        if self.response_format not in {"json_object", "json_schema", "none"}:
            raise ValueError("response_format must be json_object, json_schema, or none")

    def to_dict(self):
        """Serializable configuration contains the key's variable name, never its value."""
        return asdict(self)


class ModelCallError(RuntimeError):
    """Sanitized failure plus available evidence for caller budget accounting."""

    def __init__(self, message, *, usage=None, resolved_model=None,
                 wall_seconds=0.0, call_attempted=False):
        super().__init__(message)
        self.usage = dict(_UNKNOWN_USAGE if usage is None else usage)
        self.resolved_model = resolved_model
        self.wall_seconds = wall_seconds
        self.call_attempted = call_attempted


def _completion_url(base_url):
    if not isinstance(base_url, str) or any(char.isspace() for char in base_url):
        raise ValueError("The model endpoint must be an HTTP(S) URL")
    try:
        parsed = urlsplit(base_url.strip())
        # Reading port validates malformed ports without echoing URL credentials.
        parsed.port
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and parsed.username is None and parsed.password is None
                 and not parsed.query and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("The model endpoint must be HTTP(S), without embedded credentials, query, or fragment")
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _endpoint_url(config):
    if config.protocol == "openai_compatible":
        return _completion_url(config.base_url)
    # Reuse credential/query/control validation before checking the AWS origin.
    _completion_url(config.base_url)
    parsed = urlsplit(config.base_url)
    # the RSI-PDE Bench metering proxy (pde_rsi.bench.proxy) serves Bedrock on a loopback port
    proxy = parsed.scheme == "http" and _is_loopback(config.base_url) and not parsed.path.rstrip("/")
    if not proxy and (parsed.scheme != "https" or parsed.port not in {None, 443}
                      or parsed.path.rstrip("/")
                      or not re.fullmatch(r"bedrock-runtime(?:-fips)?\.[a-z0-9-]+\.amazonaws\.com(?:\.cn)?",
                                          parsed.hostname)):
        raise ValueError("Bedrock Converse requires a bare HTTPS AWS bedrock-runtime regional endpoint")
    if not isinstance(config.model, str) or not config.model.strip():
        raise ValueError("A model identifier is required")
    return urlunsplit((parsed.scheme, parsed.netloc,
                      "/model/" + quote(config.model, safe="") + "/converse", "", ""))


def _request_body(config, messages, schema):
    if config.response_format == "json_schema" and not isinstance(schema, dict):
        raise ValueError("json_schema response format requires a schema object")
    if config.protocol == "openai_compatible":
        body = {"model": config.model, "messages": messages, "max_tokens": config.max_output_tokens}
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif config.response_format == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "pde_rsi_proposal", "strict": True, "schema": schema}}
        return body
    if not isinstance(messages, list) or not messages:
        raise ValueError("Bedrock Converse requires a nonempty message list")
    system, conversation = [], []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Bedrock Converse input messages must contain plain text")
        role = message.get("role")
        blocks = [{"text": message["content"]}]
        if role == "system":
            system.extend(blocks)
        elif role in {"user", "assistant"}:
            conversation.append({"role": role, "content": blocks})
        else:
            raise ValueError("Bedrock Converse supports system, user, and assistant roles")
    if not conversation:
        raise ValueError("Bedrock Converse requires a conversational message")
    inference = {"maxTokens": config.max_output_tokens}
    if config.temperature is not None:
        inference["temperature"] = config.temperature
    body = {"messages": conversation, "inferenceConfig": inference}
    if system:
        body["system"] = system
    if config.response_format == "json_schema":
        body["outputConfig"] = {"textFormat": {"type": "json_schema", "structure": {
            "jsonSchema": {"name": "pde_rsi_proposal", "schema": json.dumps(schema, allow_nan=False)}}}}
    # Converse has no json_object mode. JSON instructions are supplied by the caller.
    return body


def _is_loopback(url):
    host = urlsplit(url).hostname
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _redact(value, secret):
    """Remove the configured credential even when an endpoint echoes it."""
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    return value


def _write_json(directory, name, value, secret):
    with project_path(directory / name).open("x", encoding="utf-8") as stream:
        json.dump(_redact(value, secret), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _usage(response, protocol="openai_compatible"):
    raw = response.get("usage")
    normalized = dict(_UNKNOWN_USAGE)
    if isinstance(raw, dict):
        names = {"prompt_tokens": "inputTokens", "completion_tokens": "outputTokens",
                 "total_tokens": "totalTokens"} if protocol == "bedrock_converse" else {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw.get(names.get(key, key))
            if type(value) is int and value >= 0:
                normalized[key] = value
        if protocol == "bedrock_converse" and normalized["prompt_tokens"] is not None:
            # Some models report most input as prompt-cache reads or writes; all of it is model input.
            cached = {name: raw.get(name) for name in ("cacheReadInputTokens", "cacheWriteInputTokens")}
            cached = {name: value for name, value in cached.items() if type(value) is int and value > 0}
            if cached:
                normalized["prompt_tokens"] += sum(cached.values())
                normalized["cache_tokens"] = cached
        normalized["available"] = any(normalized[key] is not None for key in
                                      ("prompt_tokens", "completion_tokens", "total_tokens"))
    return normalized


def _reject_nonfinite(value):
    raise ValueError("nonfinite_json")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite_json")
    return number


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_object_key")
        result[key] = value
    return result


def _strict_json(content):
    return json.loads(content, parse_constant=_reject_nonfinite, parse_float=_finite_float,
                      object_pairs_hook=_unique_object)


def _has_conflicting_object(prose, proposal):
    # Scan text outside the selected fence without accepting another proposal.
    # Canonical JSON comparison distinguishes values such as true and 1.
    decoder = json.JSONDecoder()
    expected = json.dumps(proposal, sort_keys=True, allow_nan=False)
    position = 0
    while True:
        position = prose.find("{", position)
        if position < 0:
            return False
        try:
            value, consumed = decoder.raw_decode(prose[position:])
        except (json.JSONDecodeError, ValueError):
            position += 1
            continue
        if isinstance(value, dict):
            try:
                actual = json.dumps(value, sort_keys=True, allow_nan=False)
            except ValueError:
                return True
            if actual != expected:
                return True
        position += consumed


def _parse_proposal(content):
    if isinstance(content, list):
        if not all(isinstance(part, dict) and isinstance(part.get("text"), str)
                   and part.get("type", "text") == "text" for part in content):
            raise ValueError("missing_content")
        content = "".join(part["text"] for part in content)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("missing_content")
    content = content.strip()
    try:
        proposal = _strict_json(content)
    except (json.JSONDecodeError, ValueError):
        fences = content.count("```")
        if fences > 2:
            raise ValueError("ambiguous_content") from None
        if fences != 2:
            raise ValueError("malformed_content") from None
        match = re.search(r"(?m)^[ \t]*```(?:(?i:json))?[ \t]*\r?\n"
                          r"(?P<body>[\s\S]*?)\r?\n[ \t]*```[ \t]*(?:\r?\n|$)", content)
        if match is None:
            raise ValueError("malformed_content") from None
        try:
            proposal = _strict_json(match.group("body").strip())
        except (json.JSONDecodeError, ValueError):
            raise ValueError("malformed_content") from None
        if not isinstance(proposal, dict):
            raise ValueError("non_object_proposal")
        outside = content[:match.start()] + "\n" + content[match.end():]
        if _has_conflicting_object(outside, proposal):
            raise ValueError("ambiguous_content")
    if not isinstance(proposal, dict):
        raise ValueError("non_object_proposal")
    return proposal


class ModelBackend:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.url = _endpoint_url(config)

    def complete(self, messages, schema, directory: Path, timeout: float):
        """Return proposal, provider usage, resolved_model, and wall_seconds.

        ``directory`` must be a fresh per-call project directory. No fallbacks,
        retries, token estimates, or model substitutions happen here.
        """
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("The model timeout must be positive and finite")
        directory = project_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if any((directory / name).exists() for name in ("request.json", "response.json")):
            raise ValueError("Use a fresh model-call directory to preserve audit files")
        key = ""
        start = time.monotonic()
        request_body = _request_body(self.config, messages, schema)
        # Validate serialization before creating an audit file or making a call.
        request_bytes = json.dumps(request_body, allow_nan=False).encode("utf-8")
        audit_request = request_body
        if self.config.protocol == "bedrock_converse":
            # The model ID lives in the route rather than the Converse body.
            audit_request = {"protocol": self.config.protocol, "requested_model": self.config.model,
                             "url": self.url, "body": request_body}
        attempted = False
        usage = dict(_UNKNOWN_USAGE)
        resolved_model = None

        def failure(code, *, status_code=None):
            elapsed = time.monotonic() - start
            error = {"type": code}
            if status_code is not None:
                error["status_code"] = status_code
            if not (directory / "response.json").exists():
                _write_json(directory, "response.json", {"error": error, "usage": usage,
                            "resolved_model": resolved_model, "wall_seconds": elapsed,
                            "call_attempted": attempted}, key)
            message = f"Model call failed: {code}"
            if status_code is not None:
                message += f" (HTTP {status_code})"
            return ModelCallError(message, usage=usage, resolved_model=resolved_model,
                                  wall_seconds=elapsed, call_attempted=attempted)

        if self.config.api_key_file is not None:
            try:
                key = project_path(self.config.api_key_file).read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError, ValueError):
                _write_json(directory, "request.json", audit_request, "")
                raise failure("api_key_file_unavailable") from None
        else:
            key = os.environ.get(self.config.api_key_env, "").strip()
        _write_json(directory, "request.json", audit_request, key)
        if "\r" in key or "\n" in key:
            raise failure("invalid_api_key") from None
        if not key and not _is_loopback(self.url):
            raise failure("missing_api_key") from None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(self.url, data=request_bytes, headers=headers, method="POST")
        attempted = True
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            # Provider error bodies and exception strings can contain credentials.
            raise failure("http_error", status_code=exc.code) from None
        except (URLError, OSError, TimeoutError, HTTPException, ValueError):
            raise failure("transport_error") from None
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise failure("response_too_large") from None
        try:
            parsed = _strict_json(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise failure("malformed_response_json") from None
        if not isinstance(parsed, dict):
            raise failure("non_object_response") from None
        usage = _usage(parsed, self.config.protocol)
        model = parsed.get("model")
        resolved_model = model if isinstance(model, str) and model.strip() else None
        _write_json(directory, "response.json", parsed, key)
        try:
            if self.config.protocol == "bedrock_converse":
                if parsed.get("stopReason") in {"max_tokens", "guardrail_intervened", "content_filtered"}:
                    raise ValueError("incomplete_response")
                output = parsed.get("output")
                message = output.get("message") if isinstance(output, dict) else None
                if not isinstance(message, dict):
                    raise ValueError("missing_message")
                blocks = message.get("content")
                if not isinstance(blocks, list):
                    raise ValueError("missing_content")
                content = [{"type": "text", "text": block["text"]} for block in blocks
                           if isinstance(block, dict) and isinstance(block.get("text"), str)]
            else:
                choices = parsed.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise ValueError("missing_choices")
                choice = choices[0]
                if choice.get("finish_reason") in {"length", "content_filter"}:
                    raise ValueError("incomplete_response")
                message = choice.get("message")
                if not isinstance(message, dict):
                    raise ValueError("missing_message")
                content = message.get("content")
            proposal = _parse_proposal(content)
        except ValueError as exc:
            raise failure(str(exc)) from None
        return {"proposal": proposal, "usage": usage, "resolved_model": resolved_model,
                "wall_seconds": time.monotonic() - start}
