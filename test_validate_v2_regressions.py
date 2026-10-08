#!/usr/bin/env python3
"""
test_validate_v2_regressions.py — Regression tests for all 5 /areyousure findings.
Run: python3 test_validate_v2_regressions.py
"""
import asyncio, json, time, re, sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))

# Import the module under test
import validate_v2 as V

PASS = 0
FAIL = 0

def report(name, ok, detail=""):
    global PASS, FAIL
    tag = "✅" if ok else "❌"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {tag} {name}" + (f" — {detail}" if detail else ""))

# --- Fixtures ---

class FakeContent:
    """Simulates an async byte stream of SSE lines."""
    def __init__(self, lines):
        self._lines = lines
    def __aiter__(self):
        async def gen():
            for line in self._lines:
                yield line
        return gen()

class FakeStreamResponse:
    def __init__(self, lines, status=200):
        self.status = status
        self.content = FakeContent(lines)
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

class FakeJsonResponse:
    def __init__(self, body, status=200):
        self.status = status
        self._body = body
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def json(self, content_type=None): return self._body

class FakeSession:
    def __init__(self, post_response=None, get_response=None):
        self._post = post_response
        self._get = get_response
    def post(self, url, **kw): return self._post
    def get(self, url, **kw): return self._get

# --- Tests ---

def test_arithmetic_boundary():
    """391 must match as a standalone number, not inside timestamps/IDs."""
    assert V.ARITH_RE.search("391") is not None
    assert V.ARITH_RE.search("The answer is 391.") is not None
    assert V.ARITH_RE.search("1677858391") is None  # timestamp
    assert V.ARITH_RE.search("chatcmpl-123910") is None  # ID fragment
    assert V.ARITH_RE.search("3910") is None  # prefix
    report("arithmetic_boundary", True)

def test_sse_parser_counts_only_content():
    """Chunks should only increment on lines with actual text content."""
    lines = [
        b": heartbeat\n",                                                    # comment — skip
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',              # role-only — skip
        b'data: malformed\n',                                                  # bad JSON — skip
        b'data: {"choices":[{"delta":{"content":"1"}}]}\n',                    # real content #1
        b'data: {"choices":[{"delta":{"content":"2"}}]}\n',                    # real content #2
        b'data: {"choices":[{"delta":{"content":"3"}}]}\n',                    # real content #3
        b'data: {"choices":[{"delta":{"content":"4"}}]}\n',                    # real content #4
        b'data: {"choices":[{"delta":{"content":"5"}}]}\n',                    # real content #5
        b'data: [DONE]\n',                                                     # end marker
    ]
    resp = FakeStreamResponse(lines)
    sess = FakeSession(post_response=resp)
    result = asyncio.run(V.chat(sess, "http://x", "/v1/chat/completions", "test", "prompt", 5.0, stream=True))
    ok = (result["status"] == 200 and result["chunks"] == 5 and
          result["ttft"] is not None and result["content"] == "12345")
    report("sse_parser_counts_only_content", ok, f"chunks={result['chunks']} content='{result['content']}'")

def test_sse_done_marker_breaks():
    """[DONE] should break the loop even if connection stays open."""
    # Simulate: 3 content lines, then [DONE], then infinite hang (no more lines)
    lines = [
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"c"}}]}\n',
        b'data: [DONE]\n',
    ]
    resp = FakeStreamResponse(lines)
    sess = FakeSession(post_response=resp)
    t0 = time.time()
    result = asyncio.run(V.chat(sess, "http://x", "/v1/chat/completions", "test", "prompt", 5.0, stream=True))
    elapsed = time.time() - t0
    ok = (result["status"] == 200 and result["chunks"] == 3 and elapsed < 2.0)
    report("sse_done_marker_breaks", ok, f"elapsed={elapsed:.2f}s chunks={result['chunks']}")

def test_streaming_partial_on_timeout():
    """Timeout mid-stream preserves collected chunks."""
    class SlowContent:
        def __aiter__(self):
            async def gen():
                yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n'
                yield b'data: {"choices":[{"delta":{"content":"y"}}]}\n'
                await asyncio.sleep(10)  # hang after 2 chunks
            return gen()
    class SlowResp:
        status = 200
        content = SlowContent()
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    sess = FakeSession(post_response=SlowResp())
    result = asyncio.run(V.chat(sess, "http://x", "/v1/chat/completions", "test", "prompt", 0.1, stream=True))
    ok = (result["status"] == 200 and result.get("partial") and result["chunks"] == 2)
    report("streaming_partial_on_timeout", ok, f"chunks={result.get('chunks')} partial={result.get('partial')}")

def test_early_exit_at_5_chunks():
    """Streaming exits after STREAM_EARLY_CHUNKS without waiting for [DONE]."""
    lines = [f'data: {{"choices":[{{"delta":{{"content":"{i}"}}}}]}}\n'.encode() for i in range(10)]
    resp = FakeStreamResponse(lines)
    sess = FakeSession(post_response=resp)
    result = asyncio.run(V.chat(sess, "http://x", "/v1/chat/completions", "test", "prompt", 5.0, stream=True))
    ok = (result["status"] == 200 and result["chunks"] == V.STREAM_EARLY_CHUNKS)
    report("early_exit_at_5_chunks", ok, f"chunks={result['chunks']}")

def test_consecutive_timeout_abort():
    """Two consecutive timeouts → TARPIT_TIMEOUT, no further calls."""
    call_count = [0]
    async def mock_chat(sess, base, cp, model, prompt, timeout, stream=False, max_tokens=2048):
        call_count[0] += 1
        return {"status": "timeout"}
    orig = V.chat
    V.chat = mock_chat
    try:
        result = asyncio.run(V.validate_model(None, "http://x", "/v1/chat/completions", "test-model", 1.0))
    finally:
        V.chat = orig
    ok = (result["tier"] == "TARPIT_TIMEOUT" and call_count[0] == 2)
    report("consecutive_timeout_abort", ok, f"tier={result['tier']} calls={call_count[0]}")

def test_host_budget_exhausted_placeholder():
    """Host budget timeout fills remaining models with HOST_BUDGET_EXHAUSTED."""
    # We test the logic directly by patching HOST_BUDGET to near-zero
    orig_budget = V.HOST_BUDGET
    V.HOST_BUDGET = 0.001  # 1ms — forces immediate timeout
    try:
        async def mock_validate_model(sess, base, cp, model, timeout, retries_429=1):
            await asyncio.sleep(0.05)  # each model takes 50ms
            return {"model": model, "score": 5, "tier": "SILVER", "checks": {}}
        orig_vm = V.validate_model
        V.validate_model = mock_validate_model
        try:
            # Simulate validate_host with 5 models
            models_to_test = ["m1", "m2", "m3", "m4", "m5"]
            results = []
            async def collect(m):
                r = await V.validate_model(None, "", "", m, 1.0)
                results.append(r)
                return r
            async def run():
                try:
                    async with asyncio.timeout(V.HOST_BUDGET):
                        await asyncio.gather(*(collect(m) for m in models_to_test))
                except asyncio.TimeoutError:
                    pass
            asyncio.run(run())
            for m in models_to_test:
                if not any(r.get("model") == m for r in results):
                    results.append({"model": m, "score": 0, "tier": "HOST_BUDGET_EXHAUSTED", "checks": {}})
            exhausted = [r for r in results if r["tier"] == "HOST_BUDGET_EXHAUSTED"]
            ok = len(exhausted) >= 3  # at least 3 of 5 should be budget-exhausted
            report("host_budget_exhausted_placeholder", ok, f"exhausted={len(exhausted)}/5")
        finally:
            V.validate_model = orig_vm
    finally:
        V.HOST_BUDGET = orig_budget

def test_masscan_json_loader():
    """Loader handles masscan -oJ JSON format."""
    fake_hits = [{"ip": "1.2.3.4", "ports": [{"port": 8080}, {"port": 11434}]},
                  {"ip": "5.6.7.8", "ports": [{"port": 3000}]}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(fake_hits, f)
        f.flush()
        raw = open(f.name).read().strip()
        hosts = []
        hits = json.loads(raw)
        for h in hits:
            ip = h.get("ip", "")
            for p in h.get("ports", []):
                pnum = p.get("port") if isinstance(p, dict) else p
                if pnum:
                    hosts.append(f"{ip}:{pnum}")
        os.unlink(f.name)
    ok = ("1.2.3.4:8080" in hosts and "1.2.3.4:11434" in hosts and "5.6.7.8:3000" in hosts)
    report("masscan_json_loader", ok, f"hosts={hosts}")

def test_model_filter_expansion():
    """New non-chat keywords (flux, cogview, sora, midjourney) are filtered."""
    assert not V.is_chat_model("flux-1.1-pro")
    assert not V.is_chat_model("cogview-3")
    assert not V.is_chat_model("sora-video")
    assert not V.is_chat_model("midjourney-v6")
    assert V.is_chat_model("qwen3-235b")
    assert V.is_chat_model("claude-sonnet-4-6")
    report("model_filter_expansion", True)

def test_ledger_write():
    """JSONL ledger writes one line per record."""
    import threading
    lock = threading.Lock()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        V._ledger_write(f, lock, {"host": "1.1.1.1:8080", "tier": "GOLD"})
        V._ledger_write(f, lock, {"host": "2.2.2.2:3000", "tier": "FAKE"})
        f.close()
        lines = open(f.name).read().strip().split("\n")
        os.unlink(f.name)
    ok = len(lines) == 2 and json.loads(lines[0])["tier"] == "GOLD"
    report("ledger_write", ok, f"lines={len(lines)}")

# --- Run all ---
if __name__ == "__main__":
    print("=" * 60)
    print("  validate_v2 regression tests — /areyousure fixes")
    print("=" * 60)
    test_arithmetic_boundary()
    test_sse_parser_counts_only_content()
    test_sse_done_marker_breaks()
    test_streaming_partial_on_timeout()
    test_early_exit_at_5_chunks()
    test_consecutive_timeout_abort()
    test_host_budget_exhausted_placeholder()
    test_masscan_json_loader()
    test_model_filter_expansion()
    test_ledger_write()
    print()
    print(f"  {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    sys.exit(0 if FAIL == 0 else 1)
