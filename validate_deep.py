#!/usr/bin/env python3
"""
validate_deep.py — 10-Prompt Deep Validator (v3).

Second-stage validator for hosts that already passed the fast battery in
validate_v2.py. Reuses validate_v2's transport (chat/discovery/ledger) and
replaces the 4-prompt soft battery with a 10-prompt deep battery that mixes
verifiable-answer prompts (arithmetic, multi-step math, recall, instruction
format, word reversal) with generation prompts and code prompts.

Scoring (max ~19):
  +1 per prompt that returns real non-stub content
  +1 bonus on each verifiable prompt whose answer is provably correct
  +2 variance (two distinct replies to the same prompt)
  +2 streaming (multi-chunk with early first token)

Tiers:  GOLD >= 13   SILVER >= 8   BRONZE >= 4   FAKE < 4

Usage:
    python3 validate_deep.py --hosts "ip:port,..." [--models 0] [--workers 8] [--timeout 12] [-o out.json]
"""
import argparse, asyncio, json, re, os, sys, time, threading
from collections import Counter

import aiohttp

# Reuse the proven transport + helpers from v2
from validate_v2 import (
    chat, check_stub, is_chat_model, probe_path_fast, _ledger_write,
    CANDIDATE_PATHS, CONNECT_TIMEOUT, HOST_BUDGET,
)

# ---- 10-prompt deep battery -------------------------------------------------
# (name, prompt, verifier_or_None)  verifier(text)->bool|None
def _has_code(t):
    return ("def " in t) or ("```" in t) or ("return" in t)

def _v_fizzbuzz(t):
    low = t.lower()
    return "fizz" in low and ("def" in low or "return" in low)

def _v_riddle(t):
    # "10 sheep, all but 7 run away" -> 7
    return bool(re.search(r"(?<!\d)7(?!\d)", t)) and not re.search(r"(?<!\d)3(?!\d)", t)

def _v_palindrome(t):
    return _has_code(t) and ("== " in t or "==s" in t.replace(" ", "") or "[::-1]" in t or "reverse" in t.lower())

def _v_arith(t):
    return bool(re.search(r"(?<!\d)391(?!\d)", t))

def _v_reverse_words(t):
    return "fox brown quick the" in re.sub(r"[^\w\s]", "", t.lower())

def _v_three_words(t):
    words = [w for w in re.split(r"\s+", t.strip()) if w]
    return len(words) == 3 and all(w.isalpha() and w.islower() for w in words)

def _v_train(t):
    return bool(re.search(r"(?<!\d)150(?!\d)", t))

def _v_gold(t):
    return bool(re.search(r"(?<![A-Za-z])Au(?![a-z])", t))

def _v_bugfix(t):
    return _has_code(t) and ("range" in t) and ("len" in t)

DEEP_PROMPTS = [
    ("fizzbuzz", "Write a short Python function fizzbuzz(n) that returns 'Fizz', 'Buzz', 'FizzBuzz', or str(n). Output code only.", _v_fizzbuzz),
    ("logic_riddle", "If I have 10 sheep and all but 7 run away, how many sheep do I have left? Respond with just the single number.", _v_riddle),
    ("code_palindrome", "Write a Python function is_palindrome(s) returning a boolean. Output code only.", _v_palindrome),
    ("creative_short", "Write a 1-sentence thought about the ocean at night.", None),
    ("arithmetic", "What is 17*23? Respond with just the number.", _v_arith),
    ("reverse_words", "Reverse the order of the words in this sentence and reply with only the result: the quick brown fox", _v_reverse_words),
    ("instruction_format", "Reply with exactly three words, all lowercase letters, no punctuation, and nothing else.", _v_three_words),
    ("multi_step_math", "A train travels at 60 km/h for 2.5 hours. How far does it travel? Reply with just the number.", _v_train),
    ("knowledge_recall", "What is the chemical symbol for gold? Reply with just the symbol.", _v_gold),
    ("code_bug_fix", "Fix this Python so it prints 0..4: `for i in range(5): print(nums[i])` where nums has 5 items. Output the corrected code only.", _v_bugfix),
]
VARIANCE_PROMPT = "Write a 1-sentence thought about the ocean at night."
STREAM_PROMPT = "Count from 1 to 30, one number per line."
TTFT_CEILING = 6.0
STREAM_EARLY_CHUNKS = 5


async def validate_model_deep(session, base, chat_path, model, timeout):
    checks, score = {}, 0.0

    # ---- Prompt 1 with fast-snap + tarpit abort ----
    n1, p1, v1 = DEEP_PROMPTS[0]
    r1 = await chat(session, base, chat_path, model, p1, timeout)
    st1 = r1.get("status") if isinstance(r1, dict) else "error"

    if isinstance(st1, int) and (400 <= st1 < 500 or st1 in (502, 503)):
        checks[n1] = f"fast_snap_{st1}"
        return {"model": model, "score": 0, "tier": f"SNAPPED_{st1}", "checks": checks}

    if st1 == "timeout":
        n2, p2, _ = DEEP_PROMPTS[1]
        r2 = await chat(session, base, chat_path, model, p2, timeout)
        if (r2.get("status") if isinstance(r2, dict) else None) == "timeout":
            checks["early_abort"] = "consecutive_timeouts"
            return {"model": model, "score": 0, "tier": "TARPIT_TIMEOUT", "checks": checks}

    # ---- Run all 10 prompts ----
    for name, prompt, verify in DEEP_PROMPTS:
        res = await chat(session, base, chat_path, model, prompt, timeout, max_tokens=400)
        st = res.get("status") if isinstance(res, dict) else "error"
        if st == 200:
            content = res.get("content", "") if isinstance(res, dict) else ""
            if content.strip() and not check_stub(content):
                score += 1
                checks[name] = "pass"
                if verify is not None:
                    try:
                        if verify(content):
                            score += 1
                            checks[name + "_correct"] = "pass"
                        else:
                            checks[name + "_correct"] = "wrong"
                    except Exception:
                        checks[name + "_correct"] = "verify_error"
                if "<thinking>" in content or (isinstance(res, dict) and res.get("raw_msg", {}).get("reasoning_content")):
                    checks["thinking_tag"] = True
                    score += 1
            else:
                checks[name] = "stub_or_empty"
        elif st == 429:
            checks[name] = "429_active_relay"
            score += 0.5
        elif isinstance(st, int) and (400 <= st < 500 or st in (502, 503)):
            checks[name] = f"4xx_{st}"
            return {"model": model, "score": round(score, 1), "tier": f"SNAPPED_{st}", "checks": checks}
        else:
            checks[name] = f"fail_{st}"

    # ---- Variance ----
    a = await chat(session, base, chat_path, model, VARIANCE_PROMPT, timeout, max_tokens=120)
    b = await chat(session, base, chat_path, model, VARIANCE_PROMPT, timeout, max_tokens=120)
    if a.get("status") == 200 and b.get("status") == 200:
        ca, cb = a.get("content", "").strip(), b.get("content", "").strip()
        if ca and cb and ca != cb:
            score += 2; checks["variance"] = "pass"
        elif ca and ca == cb:
            checks["variance"] = "IDENTICAL_REPLAY_SUSPECT"

    # ---- Streaming ----
    rs = await chat(session, base, chat_path, model, STREAM_PROMPT, timeout * 1.5, stream=True, max_tokens=300)
    if rs.get("status") == 200:
        try:
            chunks = int(rs.get("chunks", 0)); ttft = float(rs["ttft"]) if rs.get("ttft") is not None else None
            elapsed = float(rs.get("elapsed", 1.0))
        except (ValueError, TypeError):
            chunks, ttft, elapsed = 0, None, 1.0
        if chunks > 3 and ttft is not None and ttft < elapsed * 0.95 and ttft <= TTFT_CEILING:
            score += 2
            tag = " (partial)" if rs.get("partial") else ""
            checks["streaming"] = f"pass(ttft={ttft:.2f}s chunks={chunks}{tag})"
        else:
            checks["streaming"] = f"ONE_SHOT_CANNED_SUSPECT(chunks={chunks})"
    elif rs.get("status") == 429:
        checks["streaming"] = "429_active_relay"; score += 1

    tier = ("GOLD" if score >= 13 else "SILVER" if score >= 8
            else "BRONZE" if score >= 4 else "FAKE")
    return {"model": model, "score": round(score, 1), "tier": tier, "checks": checks}


async def validate_host_deep(session, host, timeout, max_models, sem, ledger_fp=None, ledger_lock=None):
    if ":" not in host:
        r = {"host": host, "tier": "DEAD", "error": "invalid_format", "working_models": []}
        _ledger_write(ledger_fp, ledger_lock, r); return r
    ip, port_str = host.rsplit(":", 1)
    try:
        port = int(port_str)
    except ValueError:
        r = {"host": host, "tier": "DEAD", "error": "invalid_port", "working_models": []}
        _ledger_write(ledger_fp, ledger_lock, r); return r

    base = f"http://{ip}:{port}"
    t0 = time.time()
    async with sem:
        discovered = None
        for res in await asyncio.gather(*(probe_path_fast(session, base, m, c) for m, c in CANDIDATE_PATHS)):
            if res:
                discovered = res; break
        if not discovered:
            r = {"host": host, "tier": "DEAD", "error": "no_models_endpoint_responsive", "working_models": []}
            _ledger_write(ledger_fp, ledger_lock, r); return r

        d_path, d_chat, raw_models = discovered
        chat_models = [m for m in raw_models if is_chat_model(m)] or raw_models
        # Deep pass: default to ALL models for a small host set
        models_to_test = chat_models if max_models <= 0 else chat_models[:max_models]

        model_sem = asyncio.Semaphore(2)  # deeper battery => fewer parallel per router
        results = []
        try:
            async with asyncio.timeout(HOST_BUDGET * 2):
                async def collect(m):
                    async with model_sem:
                        r = await validate_model_deep(session, base, d_chat, m, timeout)
                    results.append(r); return r
                await asyncio.gather(*(collect(m) for m in models_to_test))
        except asyncio.TimeoutError:
            for m in models_to_test:
                if not any(r.get("model") == m for r in results):
                    results.append({"model": m, "score": 0, "tier": "HOST_BUDGET_EXHAUSTED", "checks": {}})

        working = [r for r in results if r["tier"] in ("GOLD", "SILVER")]
        snapped = [r for r in results if r["tier"].startswith("SNAPPED_")]
        if any(r["tier"] == "GOLD" for r in results):
            host_tier = "GOLD"
        elif working:
            host_tier = "SILVER"
        elif snapped and len(snapped) == len(results):
            host_tier = "SNAPPED_4XX_HOST"
        else:
            host_tier = "FAKE"

        out = {
            "host": host, "ip": ip, "port": port,
            "endpoint_path": d_path, "chat_path": d_chat, "tier": host_tier,
            "total_models": len(raw_models), "tested_models_count": len(results),
            "working_models_count": len(working), "working_models": working,
            "all_results": results, "elapsed_s": round(time.time() - t0, 2),
            "validator": "deep_v3_10prompt",
        }
        if host_tier in ("GOLD", "SILVER"):
            flag = "👑" if host_tier == "GOLD" else "🥈"
            print(f"{flag} {host:22} {host_tier:8} working={len(working)}/{len(raw_models)}", flush=True)
        _ledger_write(ledger_fp, ledger_lock, out)
        return out


async def main():
    ap = argparse.ArgumentParser(description="10-Prompt Deep Validator (v3)")
    ap.add_argument("--hosts", required=True)
    ap.add_argument("--models", type=int, default=0, help="Max models per host (0=ALL, default)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=12.0)
    ap.add_argument("-o", "--out", default="validate_deep_results.json")
    args = ap.parse_args()

    hi = args.hosts.strip()
    if os.path.isfile(hi):
        raw = open(hi).read().strip()
        if raw.startswith("["):
            hosts = []
            for h in json.loads(raw):
                for p in h.get("ports", []):
                    pn = p.get("port") if isinstance(p, dict) else p
                    if pn: hosts.append(f"{h.get('ip','')}:{pn}")
        else:
            hosts = [l.strip() for l in raw.splitlines() if l.strip() and not l.startswith("#")]
    else:
        hosts = [h.strip() for h in hi.replace("\n", ",").split(",") if h.strip()]

    print(f"🔎 Deep 10-prompt validator on {len(hosts)} hosts, {args.workers} workers (models_cap={args.models})", flush=True)
    t0 = time.time()
    ledger_path = args.out[:-5] + ".jsonl" if args.out.endswith(".json") else args.out + ".jsonl"
    ledger_fp = open(ledger_path, "a"); lock = threading.Lock()
    sem = asyncio.Semaphore(args.workers)
    conn = aiohttp.TCPConnector(limit=args.workers * 4, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:
        out = await asyncio.gather(*(validate_host_deep(session, h, args.timeout, args.models, sem, ledger_fp, lock) for h in hosts))
    elapsed = time.time() - t0
    ledger_fp.close()
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n✨ Deep pass done in {elapsed:.1f}s")
    print(f"📊 Tiers: {dict(Counter(r['tier'] for r in out))}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
