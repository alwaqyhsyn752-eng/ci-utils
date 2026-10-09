"""Regression: a tool-call-only response must NOT be scored as a stub (2026-10-09).

Rumo's question: "did you take into account the new learnings we had today?"
This is the newest one. A gateway that injects its OWN toolset replies with
`tool_calls` + EMPTY content. Before this fix `chat()` dropped tool_calls and
returned content="", which `check_stub` called a stub -> the host was scored
FAKE. That is a false negative that would hide real hosts from a whole wave.
"""
import sys, asyncio
sys.path.insert(0, "/home/rumoto/workspace/masscan-runner")
import validate_v2 as v2


class FakeResp:
    def __init__(self, body):
        self._b, self.status = body, 200
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def json(self, content_type=None): return self._b


class FakeSession:
    def __init__(self, body): self.body = body
    def post(self, *a, **k): return FakeResp(self.body)


AEMBIT_SHAPE = {
    "choices": [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "aembitmcp-get_audit_logs", "arguments": "{}"}}]},
        "finish_reason": "tool_calls"}],
    "usage": {"prompt_tokens": 1305},
    "extra_fields": {"provider_response_headers": {"Openai-Organization": "x"}},
}


def run(body, timeout=10):
    return asyncio.run(v2.chat(FakeSession(body), "http://x:1", "/v1/chat/completions",
                               "m", "hi", timeout))


def test_tool_calls_are_captured_not_dropped():
    r = run(AEMBIT_SHAPE)
    assert r["tool_calls"] == ["aembitmcp-get_audit_logs"], r
    assert r["content"] == ""
    assert r["finish_reason"] == "tool_calls"


def test_injected_sysprompt_token_count_is_captured():
    assert run(AEMBIT_SHAPE)["prompt_tokens"] == 1305


def test_owner_id_header_leak_is_flagged():
    assert run(AEMBIT_SHAPE)["header_leak"] is True


def test_plain_text_reply_is_unchanged():
    body = {"choices": [{"message": {"content": "hello there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9}}
    r = run(body)
    assert r["content"] == "hello there"
    assert r["tool_calls"] == []
    assert r["header_leak"] is False


def test_imported_persona_markers_exist():
    for m in ("kiro", "aembit", "bizalign"):
        assert m in v2.INJECTED_PERSONA_MARKERS


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("  PASS", name)
    print("all tool-call regression tests passed")
