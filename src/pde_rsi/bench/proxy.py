"""Metering proxy between an entrant harness and Amazon Bedrock: the single place where model use is counted.

The entrant gets http://127.0.0.1:<port> as its Bedrock endpoint and a dummy credential; the proxy holds the real
token, forwards each request to bedrock-runtime, relays the response unchanged (streaming included), and reads the
token usage from it. A request is refused with HTTP 429 once any cap is reached (calls, prompt tokens, output
tokens) or for a model other than the run's model. Every call is appended to a JSON-lines ledger.

Usage is read from the response, whatever the client:
    Converse (JSON)                      usage {inputTokens, outputTokens, cacheReadInputTokens, cacheWriteInputTokens}
    ConverseStream (AWS event stream)    the "metadata" event's usage
    InvokeModel / -WithResponseStream    Anthropic message usage, or amazon-bedrock-invocationMetrics, or the
                                         X-Amzn-Bedrock-*-Token-Count headers
    OpenAI-compatible /openai/v1         Responses API response.completed usage; chat completions usage
Prompt tokens = input + cache read + cache write. A call that is under every cap when it starts may finish over a
cap; the next call is refused, so a run can exceed a token cap by at most one call.
"""

import base64
from dataclasses import dataclass, field
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
import time
from urllib.parse import unquote, urlsplit

UPSTREAM = "https://bedrock-runtime.us-east-1.amazonaws.com"
CALL_PATHS = re.compile(r"^/model/(?P<model>[^/]+)/(converse|converse-stream|invoke|invoke-with-response-stream)$"
                        r"|^/openai/v1/(responses|chat/completions)$")
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "content-length", "host", "authorization", "x-amz-security-token",
       "x-amz-date", "x-amz-content-sha256", "accept-encoding"}


@dataclass
class Caps:
    calls: int
    prompt_tokens: int
    output_tokens: int

    def to_dict(self):
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens}


@dataclass
class Meter:
    caps: Caps
    model: str
    ledger: object = None  # pathlib.Path
    calls: int = 0
    refused: int = 0
    usage: dict = field(default_factory=lambda: {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0})
    lock: threading.Lock = field(default_factory=threading.Lock)

    def prompt_tokens(self):
        return self.usage["input"]+self.usage["cache_read"]+self.usage["cache_write"]

    def exhausted(self):
        if self.calls >= self.caps.calls:
            return "model-call cap reached"
        if self.prompt_tokens() >= self.caps.prompt_tokens:
            return "prompt-token cap reached"
        if self.usage["output"] >= self.caps.output_tokens:
            return "output-token cap reached"
        return None

    def start(self):
        """Reserve one call; None, or the reason it is refused."""
        with self.lock:
            reason = self.exhausted()
            if reason:
                self.refused += 1
                return reason
            self.calls += 1
            return None

    def finish(self, record):
        with self.lock:
            for key in self.usage:
                self.usage[key] += int(record["usage"].get(key) or 0)
            if self.ledger is not None:
                with open(self.ledger, "a") as handle:
                    handle.write(json.dumps(record)+"\n")

    def totals(self):
        with self.lock:
            return {"calls": self.calls, "refused": self.refused, **self.usage, "prompt_tokens": self.prompt_tokens(),
                    "caps": self.caps.to_dict(), "exhausted": self.exhausted()}


# ---------------------------------------------------------------- usage parsers
def _anthropic(usage):
    return {"input": usage.get("input_tokens"), "cache_read": usage.get("cache_read_input_tokens"),
            "cache_write": usage.get("cache_creation_input_tokens"), "output": usage.get("output_tokens")}


def _converse(usage):
    return {"input": usage.get("inputTokens"), "cache_read": usage.get("cacheReadInputTokens"),
            "cache_write": usage.get("cacheWriteInputTokens"), "output": usage.get("outputTokens")}


def _openai(usage):
    prompt = usage.get("input_tokens", usage.get("prompt_tokens")) or 0
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    return {"input": prompt-cached, "cache_read": cached, "cache_write": 0,
            "output": usage.get("output_tokens", usage.get("completion_tokens"))}


def _metrics(metrics):
    return {"input": metrics.get("inputTokenCount"), "cache_read": metrics.get("cacheReadInputTokenCount"),
            "cache_write": metrics.get("cacheWriteInputTokenCount"), "output": metrics.get("outputTokenCount")}


def _merge(into, new):
    for key, value in new.items():
        if value is not None:
            into[key] = int(value)


def usage_from_object(item, found):
    """Fold the usage in one decoded JSON object (a response body, a stream chunk, or an event) into found."""
    if not isinstance(item, dict):
        return
    if "amazon-bedrock-invocationMetrics" in item:  # authoritative totals at the end of an invoke stream
        found["metrics"] = _metrics(item["amazon-bedrock-invocationMetrics"])
    kind = item.get("type")
    if kind == "message_start":
        _merge(found["usage"], _anthropic((item.get("message") or {}).get("usage") or {}))
    elif kind == "message_delta":
        _merge(found["usage"], {k: v for k, v in _anthropic(item.get("usage") or {}).items() if v})
    elif kind in ("response.completed", "response.incomplete", "response.failed"):
        _merge(found["usage"], _openai((item.get("response") or {}).get("usage") or {}))
    elif isinstance(item.get("usage"), dict):
        usage = item["usage"]
        if "inputTokens" in usage or "outputTokens" in usage:
            _merge(found["usage"], _converse(usage))
        elif ("prompt_tokens" in usage or "input_tokens_details" in usage
              or str(item.get("object", "")).startswith(("chat", "response"))):
            _merge(found["usage"], _openai(usage))
        else:
            _merge(found["usage"], _anthropic(usage))


class EventStream:
    """Incremental decoder of the AWS event-stream framing (prelude, headers, payload, CRC)."""

    def __init__(self, found):
        self.buffer, self.found = b"", found

    def feed(self, data):
        self.buffer += data
        while len(self.buffer) >= 12:
            total = int.from_bytes(self.buffer[0:4], "big")
            if total < 16 or len(self.buffer) < total:
                return
            headers_length = int.from_bytes(self.buffer[4:8], "big")
            payload = self.buffer[12+headers_length:total-4]
            self.buffer = self.buffer[total:]
            try:
                item = json.loads(payload)
            except ValueError:
                continue
            if isinstance(item, dict) and "bytes" in item:  # invoke stream: base64 model chunk
                try:
                    item = json.loads(base64.b64decode(item["bytes"]))
                except ValueError:
                    continue
            usage_from_object(item, self.found)


class ServerSentEvents:
    def __init__(self, found):
        self.buffer, self.found = "", found

    def feed(self, data):
        self.buffer += data.decode("utf-8", "replace")
        *lines, self.buffer = self.buffer.split("\n")
        for line in lines:
            if line.startswith("data:"):
                try:
                    usage_from_object(json.loads(line[5:].strip()), self.found)
                except ValueError:
                    pass


class Body:
    def __init__(self, found):
        self.parts, self.found = [], found

    def feed(self, data):
        self.parts.append(data)

    def close(self):
        try:
            usage_from_object(json.loads(b"".join(self.parts)), self.found)
        except ValueError:
            pass


def parser_for(content_type, found):
    content_type = (content_type or "").lower()
    if "eventstream" in content_type:
        return EventStream(found)
    if "event-stream" in content_type:
        return ServerSentEvents(found)
    return Body(found)


def final_usage(found, headers):
    usage = dict(found["metrics"] or found["usage"])
    for key, header in (("input", "x-amzn-bedrock-input-token-count"), ("output", "x-amzn-bedrock-output-token-count"),
                        ("cache_read", "x-amzn-bedrock-cache-read-input-token-count"),
                        ("cache_write", "x-amzn-bedrock-cache-write-input-token-count")):
        if not usage.get(key) and headers.get(header):
            usage[key] = int(headers[header])
    return {key: int(usage.get(key) or 0) for key in ("input", "cache_read", "cache_write", "output")}


# ---------------------------------------------------------------- server
def requested_model(path, body):
    match = CALL_PATHS.match(path)
    if match and match.group("model"):
        return unquote(match.group("model"))
    try:
        return json.loads(body).get("model")
    except (ValueError, AttributeError):
        return None


def make_handler(meter, token, upstream):
    parts = urlsplit(upstream)
    host = parts.netloc
    connect = HTTPSConnection if parts.scheme == "https" else HTTPConnection  # http only for tests

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def refuse(self, status, message):
            body = json.dumps({"message": f"RSI-PDE Bench: {message}", "__type": "ThrottlingException"}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-amzn-ErrorType", "ThrottlingException")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.forward(b"")

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            path = urlsplit(self.path).path
            if CALL_PATHS.match(path):
                model = requested_model(path, body)
                if model != meter.model:
                    return self.refuse(403, f"only {meter.model} may be called in this run, not {model}")
                reason = meter.start()
                if reason:
                    return self.refuse(429, f"budget exhausted ({reason})")
                started = time.time()
                found = {"usage": {}, "metrics": None}
                status, headers = self.forward(body, found)
                meter.finish({"time": started, "seconds": round(time.time()-started, 3), "path": path, "model": model,
                              "status": status, "usage": final_usage(found, headers)})
            else:
                self.forward(body)

        def forward(self, body, found=None):
            connection = connect(host, timeout=1800)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            headers.update({"Host": parts.hostname, "Authorization": f"Bearer {token}", "Accept-Encoding": "identity"})
            if body:
                headers["Content-Length"] = str(len(body))
            try:
                connection.request(self.command, self.path, body=body or None, headers=headers)
                response = connection.getresponse()
            except OSError as error:
                self.refuse(502, f"upstream unreachable ({type(error).__name__})")
                return 502, {}
            received = {k.lower(): v for k, v in response.getheaders()}
            parser = parser_for(received.get("content-type"), found) if found is not None else None
            client = True  # False once the entrant hangs up (e.g. killed at its deadline)

            def send(data):
                nonlocal client
                if client:
                    try:
                        self.wfile.write(data)
                        self.wfile.flush()
                    except OSError:
                        client = False
            try:
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in HOP:
                        self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
            except OSError:
                client = False
            while True:  # read the whole response even without a client: its usage is still counted
                chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
                if not chunk:
                    break
                if parser is not None:
                    parser.feed(chunk)
                send(f"{len(chunk):x}\r\n".encode()+chunk+b"\r\n")
            send(b"0\r\n\r\n")
            if not client:
                self.close_connection = True
            if isinstance(parser, Body):
                parser.close()
            connection.close()
            return response.status, received

    return Handler


class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return  # a client closed its keep-alive connection; nothing was in flight
        super().handle_error(request, client_address)


class MeteringProxy:
    def __init__(self, meter, token, upstream=UPSTREAM, port=0):
        self.meter = meter
        self.server = _Server(("127.0.0.1", port), make_handler(meter, token, upstream))
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
