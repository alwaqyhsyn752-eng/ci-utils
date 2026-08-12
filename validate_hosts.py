#!/usr/bin/env python3
"""4-prompt stub validation for Ollama hosts.

Hits each host with 4 diverse prompts (2 code, 1 poem, 1 prose) and checks
the responses against known stub/refusal markers. A host passes only if all
4 prompts return substantive, non-stub responses.

Usage:
    python3 validate_hosts.py --hosts "ip1:11434,ip2:11434" --model "kimi-k2.6:cloud"
"""
import argparse
import json
import re
import sys
import time
import urllib.request

PROMPTS = [
    ("code_fizzbuzz",
     "Write a Python function called fizzbuzz that prints numbers 1 to 20. "
     "For multiples of 3 print Fizz, multiples of 5 print Buzz, both print FizzBuzz. "
     "Give only the code, no explanation."),
    ("code_dict_sort",
     "Write a Python one-liner to sort a dictionary by its values in descending order. "
     "Give only the code."),
    ("poem_ocean",
     "Write a 4-line poem about the ocean. Rhyming. No introduction."),
    ("prose_rain",
     "Write exactly 50 words of prose about someone walking home in the rain. "
     "No introduction, just the prose."),
]

STUB_MARKERS = [
    "regarding", "i'd be happy to help", "let me think about this",
    "i'll do my best to assist", "great question", "that's an interesting question",
    "based on my understanding", "here's what i can tell you",
    "i can provide some context", "i would recommend approaching",
    "i'm a helpful ai assistant", "i cannot assist",
    "as an ai language model", "i'm just an ai", "i don't have the ability",
]

MIN_LEN = 40  # substantive response threshold (chars)


def chat(host: str, model: str, prompt: str, timeout: int = 60) -> tuple[int, str]:
    """Returns (http_status, response_text)."""
    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "validator/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        if "error" in d:
            return r.status, json.dumps(d["error"])
        content = ((d.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "")
        return r.status, content or ""
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read().decode(errors="ignore")[:300]
        except Exception:
            return e.code, ""
    except Exception as e:
        return 0, str(e)[:200]


def is_stub(text: str) -> bool:
    low = text.lower()
    if len(text.strip()) < MIN_LEN:
        return True
    for marker in STUB_MARKERS:
        if marker in low:
            return True
    return False


def validate_host(host: str, model: str) -> dict:
    result = {"host": host, "model": model, "prompts": {}, "passed": True}
    for name, prompt in PROMPTS:
        status, text = chat(host, model, prompt)
        stub = is_stub(text)
        ok = status == 200 and not stub
        result["prompts"][name] = {
            "http": status,
            "stub": stub,
            "ok": ok,
            "len": len(text),
            "head": text[:120].replace("\n", " "),
        }
        if not ok:
            result["passed"] = False
        time.sleep(0.5)  # be polite
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", required=True, help="comma-separated host:port list")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    print(f"validating {len(hosts)} hosts for model={args.model}", flush=True)

    results = []
    for host in hosts:
        print(f"  → {host} ...", end=" ", flush=True)
        r = validate_host(host, args.model)
        results.append(r)
        print("PASS" if r["passed"] else "FAIL", flush=True)
        for pname, p in r["prompts"].items():
            tag = "✓" if p["ok"] else "✗"
            print(f"      {tag} {pname}: http={p['http']} len={p['len']} stub={p['stub']}", flush=True)
            if not p["ok"]:
                print(f"        head: {p['head']!r}", flush=True)

    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    passed = [r["host"] for r in results if r["passed"]]
    failed = [r["host"] for r in results if not r["passed"]]
    print(f"\n{'='*50}")
    print(f"PASSED ({len(passed)}): {passed}")
    print(f"FAILED ({len(failed)}): {failed}")
    print(f"results saved to validation_results.json")


if __name__ == "__main__":
    main()
