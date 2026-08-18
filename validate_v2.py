#!/usr/bin/env python3
"""
validate_v2.py — hardened host validator.

Upgrades over v1 (batch-21 audit findings):
  1. Determinism check      — same prompt twice @ temp 1.0; byte-identical = replay farm (HARD signal)
  2. Verifiable arithmetic  — "17*23, number only" must contain 391 (HARD signal)
  3. Streaming fingerprint  — TTFT + chunk count + tokens/sec; one-shot body = canned (HARD signal)
  4. <thinking> tag bonus   — real thinking models leak scratchpads (SOFT bonus)
  5. 429 => backoff-retry (active shared relay, NOT dead); timeout => RETAIN, never prune (policy)
  6. 4xx => snap model; all-models-4xx => snap host (policy)

Usage:
    python3 validate_v2.py --hosts "ip:port,ip:port" [--models N] [--workers 8] [--timeout 12] [-o out.json]
"""
import argparse, asyncio, json, time, sys

import aiohttp

SOFT_PROMPTS = [
    ("fizzbuzz", "Write a short Python function fizzbuzz(n) that returns 'Fizz', 'Buzz', 'FizzBuzz', or str(n). Output code only."),
    ("logic_riddle", "If I have 10 sheep and all but 7 run away, how many sheep do I have left? Respond with just the single number."),
    ("code_palindrome", "Write a Python function is_palindrome(s) returning boolean. Output code only."),
    ("creative_short", "Write a 1-sentence thought about the ocean at night."),
]
VARIANCE_PROMPT = "Write a 1-sentence thought about the ocean at night."
ARITH_PROMPT = "What is 17*23? Respond with just the number."
ARITH_ANSWER = "391"
STREAM_PROMPT = "Count from 1 to 30, one number per line."


async def chat(session, base, model, prompt, timeout, stream=False, max_tokens=2048):
    """One chat call. Returns dict(status, content, ttft, chunks, elapsed) or error info."""
    t0 = time.time()
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "stream": stream}
    try:
        async with asyncio.timeout(timeout):
            if not stream:
                async with session.post(f"{base}/v1/chat/completions", json=payload) as r:
                    if r.status != 200:
                        return {"status": r.status}
                    body = await r.json(content_type=None)
                    content = body["choices"][0]["message"].get("content") or ""
                    return {"status": 200, "content": content, "elapsed": time.time() - t0}
            # streaming
            ttft, chunks, text = None, 0, []
            async with session.post(f"{base}/v1/chat/completions", json=payload) as r:
                if r.status != 200:
                    return {"status": r.status}
                async for line in r.content:
                    line = line.strip()
                    if not line.startswith(b"data:") or line == b"data: [DONE]":
                        continue
                    if ttft is None:
                        ttft = time.time() - t0
                    chunks += 1
                    try:
                        delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content") or ""
                        text.append(delta)
                    except Exception:
                        pass
            elapsed = time.time() - t0
            return {"status": 200, "content": "".join(text), "ttft": ttft,
                    "chunks": chunks, "elapsed": elapsed}
    except asyncio.TimeoutError:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)[:100]}


async def validate_model(session, base, model, timeout, retries_429=2):
    checks, score = {}, 0
    # --- soft prompts (1 pt each, 429-aware) ---
    for name, prompt in SOFT_PROMPTS:
        for attempt in range(retries_429 + 1):
            res = await chat(session, base, model, prompt, timeout)
            if res.get("status") == 429 and attempt < retries_429:
                await asyncio.sleep(3 * (attempt + 1))
                continue
            break
        if res.get("status") == 200 and len(res["content"].strip()) > 10:
            score += 1
            checks[name] = "pass"
            if "<thinking>" in res["content"]:
                checks["thinking_tag"] = True
                score += 1  # real-thinking bonus
        elif res.get("status") == 429:
            checks[name] = "429_active_relay"
        elif res.get("status") == "timeout":
            checks[name] = "timeout_retain"
        elif isinstance(res.get("status"), int) and 400 <= res["status"] < 500:
            checks[name] = f"4xx_{res['status']}"
            return {"model": model, "score": 0, "tier": "SNAPPED_4XX", "checks": checks}
        else:
            checks[name] = "fail"
    # --- HARD: determinism (2 pts) ---
    r1 = await chat(session, base, model, VARIANCE_PROMPT, timeout, max_tokens=200)
    r2 = await chat(session, base, model, VARIANCE_PROMPT, timeout, max_tokens=200)
    if r1.get("status") == 200 and r2.get("status") == 200:
        if r1["content"].strip() != r2["content"].strip():
            score += 2
            checks["variance"] = "pass"
        else:
            checks["variance"] = "IDENTICAL_REPLAY_SUSPECT"
    # --- HARD: verifiable arithmetic (2 pts) ---
    ra = await chat(session, base, model, ARITH_PROMPT, timeout, max_tokens=50)
    if ra.get("status") == 200:
        if ARITH_ANSWER in ra["content"].replace(",", "").replace(" ", ""):
            score += 2
            checks["arithmetic"] = "pass"
        else:
            checks["arithmetic"] = f"wrong_answer:{ra['content'][:40]!r}"
    # --- HARD: streaming fingerprint (2 pts) ---
    rs = await chat(session, base, model, STREAM_PROMPT, timeout * 2, stream=True, max_tokens=300)
    if rs.get("status") == 200:
        streamed = rs.get("chunks", 0) > 3 and rs.get("ttft") is not None and rs["ttft"] < rs["elapsed"] * 0.9
        if streamed:
            score += 2
            checks["streaming"] = f"pass(ttft={rs['ttft']:.2f}s chunks={rs['chunks']})"
        else:
            checks["streaming"] = f"ONE_SHOT_CANNED_SUSPECT(chunks={rs.get('chunks')})"
    elif rs.get("status") == 429:
        checks["streaming"] = "429_active_relay"
        score += 1  # relay alive but busy — partial credit, do not condemn
    tier = "GOLD" if score >= 8 else "SILVER" if score >= 5 else "BRONZE" if score >= 3 else "FAKE"
    return {"model": model, "score": score, "tier": tier, "checks": checks}


async def validate_host(session, host, timeout, max_models, sem):
    ip, port = host.rsplit(":", 1)
    base = f"http://{ip}:{port}"
    async with sem:
        try:
            async with asyncio.timeout(timeout):
                async with session.get(f"{base}/v1/models") as r:
                    if r.status != 200:
                        return {"host": host, "tier": "DEAD", "working_models": []}
                    data = await r.json(content_type=None)
        except Exception as e:
            return {"host": host, "tier": "DEAD", "error": str(e)[:80], "working_models": []}
        models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        if not models:
            return {"host": host, "tier": "DEAD", "working_models": []}
        results = []
        for m in models[:max_models]:
            results.append(await validate_model(session, base, m, timeout))
        working = [r for r in results if r["tier"] in ("GOLD", "SILVER")]
        snapped = [r for r in results if r["tier"] == "SNAPPED_4XX"]
        host_tier = "GOLD" if any(r["tier"] == "GOLD" for r in results) else \
                    "SILVER" if working else "SNAPPED_4XX_HOST" if snapped and len(snapped) == len(results) else "FAKE"
        return {"host": host, "ip": ip, "port": int(port), "tier": host_tier,
                "total_models": len(models), "tested": len(results),
                "working_models": working, "all_results": results}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", required=True)
    ap.add_argument("--models", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=12)
    ap.add_argument("-o", "--out", default="validate_v2_results.json")
    args = ap.parse_args()

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    sem = asyncio.Semaphore(args.workers)
    conn = aiohttp.TCPConnector(limit=args.workers * 4, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as s:
        out = await asyncio.gather(*(validate_host(s, h, args.timeout, args.models, sem) for h in hosts))
    json.dump(out, open(args.out, "w"), indent=2)
    from collections import Counter
    print("host tiers:", dict(Counter(r["tier"] for r in out)))
    for r in out:
        flag = "👑" if r["tier"] == "GOLD" else "🥈" if r["tier"] == "SILVER" else "🚫" if "SNAP" in r["tier"] or r["tier"] == "FAKE" else "💀"
        sus = [c for m in r.get("all_results", []) for k, c in m["checks"].items() if "SUSPECT" in str(c)]
        print(f'{flag} {r["host"]:22} {r["tier"]:16} working={len(r["working_models"])}' + (f'  ⚠{sus[:2]}' if sus else ""))


if __name__ == "__main__":
    asyncio.run(main())
