#!/usr/bin/env python3
"""
validate_v2.py — Hardened Multi-Model Router & Gateway Validator (Zero-Ollama).

Upgrades over v1:
  1. Strict Ollama Filter     — Rejects port 11434 instantly (no Ollama stubs/honeypots).
  2. Multi-Path Discovery     — Probes /v1/models, /api/v1/models, /openai/v1/models, /model/info.
  3. Complete Catalog Testing — Tests every advertised model by default (no 3-model truncation).
  4. 10-Signal Battery:
     - 4 soft prompts (fizzbuzz, logic riddle, palindrome, creative)
     - <thinking> scratchpad tag bonus (+1 for genuine reasoning models)
     - Variance / Replay check (2 calls @ temp 1.0; byte-identical = CANNED_REPLAY_HONEYPOT)
     - Verifiable arithmetic ("17*23, number only" must contain 391)
     - Streaming TTFT & chunk metrics (TTFT < 90% total, chunks > 3; one-shot = ONE_SHOT_CANNED)
     - 429 backoff-retry (active shared relay credit, NEVER condemn busy nodes)
     - Honeypot & stub marker filtering (qyora-gate, 久辰, "as an ai language model", etc.)
     - 4xx snap isolation (isolate broken model routes, only snap host if 100% of models return 4xx)

Usage:
    python3 validate_v2.py --hosts "ip:port,ip:port" [--models 0] [--workers 8] [--timeout 12] [-o out.json]
"""
import argparse, asyncio, json, time, sys, re
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

CANDIDATE_PATHS = [
    ("/v1/models", "/v1/chat/completions"),
    ("/api/v1/models", "/api/v1/chat/completions"),
    ("/openai/v1/models", "/openai/v1/chat/completions"),
    ("/model/info", "/v1/chat/completions"),
]

STUB_MARKERS = [
    "as an ai language model", "i'm a helpful ai assistant", "i'm just an ai",
    "regarding", "i'd be happy to help", "let me think about this",
    "i'll do my best to assist", "qyora-gate", "久辰", "owner of the router",
    "i cannot fulfill this request", "token plan not support"
]

def check_stub(text: str) -> bool:
    if not text:
        return True
    low = text.lower()
    for m in STUB_MARKERS:
        if m in low:
            return True
    return False

async def chat(session, base, chat_path, model, prompt, timeout, stream=False, max_tokens=2048):
    """One chat call. Returns dict(status, content, ttft, chunks, elapsed) or error info."""
    t0 = time.time()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream
    }
    url = f"{base}{chat_path}"
    try:
        async with asyncio.timeout(timeout):
            if not stream:
                async with session.post(url, json=payload) as r:
                    if r.status != 200:
                        return {"status": r.status}
                    try:
                        body = await r.json(content_type=None)
                    except Exception:
                        return {"status": "bad_json"}
                    choices = body.get("choices") if isinstance(body, dict) else []
                    if not choices or not isinstance(choices, list):
                        return {"status": "no_choices"}
                    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                    content = msg.get("content") or "" if isinstance(msg, dict) else ""
                    reasoning = msg.get("reasoning_content") or "" if isinstance(msg, dict) else ""
                    full_text = (content or reasoning).strip()
                    return {"status": 200, "content": full_text, "raw_msg": msg, "elapsed": time.time() - t0}

            # Streaming mode
            ttft, chunks, text = None, 0, []
            async with session.post(url, json=payload) as r:
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
                        delta = json.loads(line[5:])["choices"][0].get("delta", {})
                        d_text = delta.get("content") or delta.get("reasoning_content") or ""
                        text.append(d_text)
                    except Exception:
                        pass
            elapsed = time.time() - t0
            return {
                "status": 200,
                "content": "".join(text).strip(),
                "ttft": ttft,
                "chunks": chunks,
                "elapsed": elapsed
            }
    except asyncio.TimeoutError:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)[:100]}

async def validate_model(session, base, chat_path, model, timeout, retries_429=2):
    checks, score = {}, 0

    # 1. Soft Prompts (1 pt each, 429-aware, reasoning bonus)
    for name, prompt in SOFT_PROMPTS:
        res = None
        for attempt in range(retries_429 + 1):
            res = await chat(session, base, chat_path, model, prompt, timeout)
            if res.get("status") == 429 and attempt < retries_429:
                await asyncio.sleep(2.5 * (attempt + 1))
                continue
            break

        st = res.get("status") if res else "error"
        if st == 200:
            content = res.get("content", "")
            if len(content) > 10 and not check_stub(content):
                score += 1
                checks[name] = "pass"
                if "<thinking>" in content or res.get("raw_msg", {}).get("reasoning_content"):
                    checks["thinking_tag"] = True
                    score += 1
                if name == "logic_riddle":
                    # strictly check for 7 and not 3 as dominant answer
                    if "7" in content and not ("3" in content and content.index("3") < content.index("7")):
                        score += 1
                        checks["riddle_logic"] = "pass"
                    else:
                        checks["riddle_logic"] = "failed_deduction"
            else:
                checks[name] = "stub_or_empty"
        elif st == 429:
            checks[name] = "429_active_relay"
            score += 0.5  # relay is alive and gating
        elif st == "timeout":
            checks[name] = "timeout_retain"
        elif isinstance(st, int) and 400 <= st < 500:
            checks[name] = f"4xx_{st}"
            return {"model": model, "score": 0, "tier": "SNAPPED_4XX", "checks": checks}
        else:
            checks[name] = f"fail_{st}"

    # 2. HARD: Variance / Determinism Check (2 pts)
    r1 = await chat(session, base, chat_path, model, VARIANCE_PROMPT, timeout, max_tokens=200)
    r2 = await chat(session, base, chat_path, model, VARIANCE_PROMPT, timeout, max_tokens=200)
    if r1.get("status") == 200 and r2.get("status") == 200:
        c1, c2 = r1.get("content", "").strip(), r2.get("content", "").strip()
        if c1 and c2 and c1 != c2:
            score += 2
            checks["variance"] = "pass"
        elif c1 and c1 == c2:
            checks["variance"] = "IDENTICAL_REPLAY_SUSPECT"
        else:
            checks["variance"] = "empty_responses"

    # 3. HARD: Verifiable Arithmetic Check (2 pts)
    ra = await chat(session, base, chat_path, model, ARITH_PROMPT, timeout, max_tokens=60)
    if ra.get("status") == 200:
        cleaned = ra.get("content", "").replace(",", "").replace(" ", "")
        if ARITH_ANSWER in cleaned:
            score += 2
            checks["arithmetic"] = "pass"
        else:
            checks["arithmetic"] = f"wrong_answer:{ra.get('content', '')[:30]!r}"

    # 4. HARD: Streaming Fingerprint (2 pts)
    rs = await chat(session, base, chat_path, model, STREAM_PROMPT, timeout * 1.5, stream=True, max_tokens=300)
    if rs.get("status") == 200:
        try:
            chunks = int(rs.get("chunks", 0))
            ttft = float(rs["ttft"]) if rs.get("ttft") is not None else None
            elapsed = float(rs.get("elapsed", 1.0))
        except (ValueError, TypeError):
            chunks, ttft, elapsed = 0, None, 1.0

        if chunks > 3 and ttft is not None and ttft < elapsed * 0.95:
            score += 2
            checks["streaming"] = f"pass(ttft={ttft:.2f}s chunks={chunks})"
        else:
            checks["streaming"] = f"ONE_SHOT_CANNED_SUSPECT(chunks={chunks})"
    elif rs.get("status") == 429:
        checks["streaming"] = "429_active_relay"
        score += 1

    tier = "GOLD" if score >= 8 else "SILVER" if score >= 5 else "BRONZE" if score >= 3 else "FAKE"
    return {"model": model, "score": score, "tier": tier, "checks": checks}

async def validate_host(session, host, timeout, max_models, sem):
    if ":" not in host:
        return {"host": host, "tier": "DEAD", "error": "invalid_host_format", "working_models": []}

    ip, port_str = host.rsplit(":", 1)
    port = int(port_str)

    # 1. HARD FILTER: Ollama is completely excluded
    if port == 11434:
        return {"host": host, "tier": "FILTERED_OLLAMA", "error": "ollama_excluded", "working_models": []}

    base = f"http://{ip}:{port}"
    async with sem:
        discovered_path = None
        discovered_chat = None
        models = []

        # 2. Multi-Path Probe
        for m_path, c_path in CANDIDATE_PATHS:
            try:
                async with asyncio.timeout(timeout):
                    async with session.get(f"{base}{m_path}") as r:
                        if r.status == 200:
                            try:
                                data = await r.json(content_type=None)
                            except Exception:
                                continue
                            
                            # Parse models
                            m_list = []
                            if isinstance(data, dict):
                                raw_m = data.get("data") or data.get("models") or []
                                if isinstance(raw_m, list):
                                    for item in raw_m:
                                        if isinstance(item, dict):
                                            mid = item.get("id") or item.get("name")
                                            if mid: m_list.append(str(mid))
                                        elif isinstance(item, str):
                                            m_list.append(item)
                                elif isinstance(raw_m, dict): # litellm /model/info dict
                                    m_list.extend([str(k) for k in raw_m.keys()])
                                elif m_path == "/model/info":
                                    m_list.extend([str(k) for k in data.keys() if k != "data"])

                            if m_list:
                                discovered_path = m_path
                                discovered_chat = c_path
                                models = list(dict.fromkeys(m_list)) # dedupe preserving order
                                break
            except Exception:
                pass

        if not models:
            return {"host": host, "tier": "DEAD", "error": "no_models_endpoint_responsive", "working_models": []}

        # 3. Test every available model (or up to max_models if specified > 0)
        models_to_test = models if max_models <= 0 else models[:max_models]
        results = []
        for m in models_to_test:
            res = await validate_model(session, base, discovered_chat, m, timeout)
            results.append(res)

        working = [r for r in results if r["tier"] in ("GOLD", "SILVER")]
        snapped = [r for r in results if r["tier"] == "SNAPPED_4XX"]

        if any(r["tier"] == "GOLD" for r in results):
            host_tier = "GOLD"
        elif working:
            host_tier = "SILVER"
        elif snapped and len(snapped) == len(results):
            host_tier = "SNAPPED_4XX_HOST"
        else:
            host_tier = "FAKE"

        return {
            "host": host,
            "ip": ip,
            "port": port,
            "endpoint_path": discovered_path,
            "chat_path": discovered_chat,
            "tier": host_tier,
            "total_models": len(models),
            "tested_models_count": len(results),
            "working_models_count": len(working),
            "working_models": working,
            "all_results": results
        }

async def main():
    ap = argparse.ArgumentParser(description="v2 Hardened Multi-Model Router Validator (Zero-Ollama)")
    ap.add_argument("--hosts", required=True, help="Comma-separated host:port list or path to a hosts file")
    ap.add_argument("--models", type=int, default=0, help="Max models to test per host (0 = test ALL models)")
    ap.add_argument("--workers", type=int, default=12, help="Concurrent host workers")
    ap.add_argument("--timeout", type=float, default=12.0, help="Per-request timeout seconds")
    ap.add_argument("-o", "--out", default="validate_v2_results.json", help="Output JSON path")
    args = ap.parse_args()

    # Support passing file path or comma-separated string
    hosts_input = args.hosts.strip()
    if sys.platform != "win32" and ("\n" in hosts_input or "," in hosts_input or ":" in hosts_input):
        import os
        if os.path.isfile(hosts_input):
            with open(hosts_input) as fp:
                hosts = [line.strip() for line in fp if line.strip() and not line.startswith("#")]
        else:
            hosts = [h.strip() for h in hosts_input.replace("\n", ",").split(",") if h.strip()]
    else:
        import os
        if os.path.isfile(hosts_input):
            with open(hosts_input) as fp:
                hosts = [line.strip() for line in fp if line.strip() and not line.startswith("#")]
        else:
            hosts = [h.strip() for h in hosts_input.split(",") if h.strip()]

    # Filter out Ollama port 11434 before dispatching
    filtered_hosts = [h for h in hosts if not h.endswith(":11434")]
    dropped_ollama = len(hosts) - len(filtered_hosts)
    if dropped_ollama > 0:
        print(f"🛡️ Pre-filtered {dropped_ollama} Ollama hosts (port 11434). Remaining: {len(filtered_hosts)}")

    sem = asyncio.Semaphore(args.workers)
    conn = aiohttp.TCPConnector(limit=args.workers * 4, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:
        out = await asyncio.gather(*(validate_host(session, h, args.timeout, args.models, sem) for h in filtered_hosts))

    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=2)

    from collections import Counter
    counts = dict(Counter(r["tier"] for r in out))
    print("\n📊 Host Tiers Summary:", counts)
    for r in out:
        if r["tier"] in ("GOLD", "SILVER"):
            flag = "👑" if r["tier"] == "GOLD" else "🥈"
            print(f'{flag} {r["host"]:24} {r["tier"]:10} models_working={r["working_models_count"]}/{r["total_models"]}')

if __name__ == "__main__":
    asyncio.run(main())
