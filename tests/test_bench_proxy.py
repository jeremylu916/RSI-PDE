"""The metering proxy relays requests, reads usage from every response format, and enforces the caps."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pde_rsi.bench.proxy import Caps, Meter, MeteringProxy

MODEL = "us.anthropic.claude-opus-5-5"


def frame(item):
    payload = json.dumps({"bytes": base64.b64encode(json.dumps(item).encode()).decode()}).encode()
    total = 12+len(payload)+4
    return total.to_bytes(4, "big")+(0).to_bytes(4, "big")+b"\0"*4+payload+b"\0"*4


class Upstream(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        Upstream.seen.append((self.path, self.headers.get("Authorization")))
        if self.path.endswith("/converse"):
            body, kind = json.dumps({"output": {}, "usage": {"inputTokens": 100, "outputTokens": 7,
                                                            "cacheReadInputTokens": 50}}).encode(), "application/json"
        elif self.path.endswith("/invoke-with-response-stream"):
            body = (frame({"type": "message_start", "message": {"usage": {"input_tokens": 3, "cache_read_input_tokens": 900,
                                                                          "cache_creation_input_tokens": 20, "output_tokens": 1}}})
                    + frame({"type": "message_delta", "usage": {"output_tokens": 42}})
                    + frame({"type": "message_stop", "amazon-bedrock-invocationMetrics": {
                        "inputTokenCount": 3, "outputTokenCount": 42, "cacheReadInputTokenCount": 900,
                        "cacheWriteInputTokenCount": 20}}))
            kind = "application/vnd.amazon.eventstream"
        else:
            completed = {"type": "response.completed", "response": {"usage": {
                "input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600}, "output_tokens": 30}}}
            body = (b"event: response.created\ndata: {\"type\": \"response.created\"}\n\n"
                    + b"event: response.completed\ndata: "+json.dumps(completed).encode()+b"\n\n")
            kind = "text/event-stream"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def post(url, path, payload=b"{}"):
    request = Request(url+path, data=payload, headers={"Content-Type": "application/json",
                                                        "Authorization": "Bearer dummy"}, method="POST")
    with urlopen(request, timeout=30) as response:
        return response.status, response.read()


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.directory = Path(tempfile.mkdtemp())
        Upstream.seen.clear()

    def tearDown(self):
        self.upstream.shutdown()
        self.upstream.server_close()

    def proxy(self, caps):
        meter = Meter(caps, MODEL, ledger=self.directory/"ledger.jsonl")
        return meter, MeteringProxy(meter, "real-token", upstream=f"http://127.0.0.1:{self.upstream.server_address[1]}")

    def test_usage_of_each_format_and_credential_swap(self):
        meter, proxy = self.proxy(Caps(10, 10**9, 10**9))
        with proxy:
            status, body = post(proxy.url, f"/model/{MODEL}/converse")
            self.assertEqual(status, 200)
            self.assertIn(b"inputTokens", body)
            _, stream = post(proxy.url, f"/model/{MODEL}/invoke-with-response-stream")
            self.assertEqual(stream.count(b"bytes"), 3)  # relayed unchanged
            post(proxy.url, "/openai/v1/responses", json.dumps({"model": MODEL}).encode())
        totals = meter.totals()
        self.assertEqual(totals["calls"], 3)
        self.assertEqual((totals["input"], totals["cache_read"], totals["cache_write"], totals["output"]),
                         (100+3+400, 50+900+600, 20, 7+42+30))
        self.assertTrue(all(auth == "Bearer real-token" for _, auth in Upstream.seen))
        self.assertEqual(len((self.directory/"ledger.jsonl").read_text().splitlines()), 3)

    def test_caps_and_model_are_enforced(self):
        meter, proxy = self.proxy(Caps(2, 10**9, 45))
        with proxy:
            post(proxy.url, f"/model/{MODEL}/invoke-with-response-stream")  # 42 output tokens
            post(proxy.url, f"/model/{MODEL}/converse")                     # starts under the cap, ends at 49
            with self.assertRaises(HTTPError) as refused:
                post(proxy.url, f"/model/{MODEL}/converse")
            self.assertEqual(refused.exception.code, 429)
            with self.assertRaises(HTTPError) as other:
                post(proxy.url, "/model/us.openai.gpt-5.5/converse")
            self.assertEqual(other.exception.code, 403)
        self.assertEqual(meter.totals()["calls"], 2)
        self.assertEqual(len(Upstream.seen), 2)


if __name__ == "__main__":
    unittest.main()
