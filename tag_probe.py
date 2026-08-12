#!/usr/bin/env python3
"""Tag-probe masscan hits for Ollama :cloud models.

Reads hits.json (masscan output), probes each IP's /api/tags,
extracts all :cloud models, and outputs a structured result.

Usage: python3 tag_probe.py <hits.json> <output.json>
"""
import json, sys, time, urllib.request, socket
from concurrent.futures import ThreadPoolExecutor, as_completed

def probe_tags(ip):
    """Fetch /api/tags from an Ollama instance, return all model names."""
    url = f"http://{ip}:11434/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Ollama-Tag-Probe/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.load(r)
        models = [m.get("name", "") for m in data.get("models", [])]
        cloud = [m for m in models if ":cloud" in m]
        return {"ip": ip, "all_models": models, "cloud_models": cloud, "status": "ok"}
    except urllib.error.HTTPError as e:
        return {"ip": ip, "error": f"HTTP {e.code}", "cloud_models": [], "status": "error"}
    except Exception as e:
        return {"ip": ip, "error": str(e)[:100], "cloud_models": [], "status": "error"}

def main():
    hits_file = sys.argv[1]
    output_file = sys.argv[2]

    # Parse masscan JSON output
    raw = open(hits_file).read().strip()
    if not raw or raw == "[]":
        print("No hits found.")
        json.dump({"hosts": [], "cloud_summary": {}}, open(output_file, "w"))
        return

    try:
        hits = json.loads(raw)
    except json.JSONDecodeError:
        # masscan outputs concatenated JSON objects, not an array
        hits = []
        for line in raw.strip().split("\n"):
            try:
                hits.append(json.loads(line))
            except:
                pass

    # Extract unique IPs with port 11434 open
    ips = set()
    for hit in hits:
        ip = hit.get("ip") or hit.get("ip_str", "")
        ports = hit.get("ports", [])
        for p in ports:
            if isinstance(p, dict) and p.get("port") == 11434:
                ips.add(ip)
                break
        if not ports and hit.get("ip"):
            ips.add(ip)

    ips = sorted(ips)
    print(f"Tag-probing {len(ips)} unique hosts with :11434 open...", flush=True)

    results = []
    cloud_hosts = {}  # model -> [ips]
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=50) as ex:
        futs = {ex.submit(probe_tags, ip): ip for ip in ips}
        done = 0
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            if res["status"] == "ok" and res["cloud_models"]:
                results.append(res)
                for m in res["cloud_models"]:
                    cloud_hosts.setdefault(m, []).append(res["ip"])
                print(f"  CLOUD-HIT {res['ip']}: {res['cloud_models']}", flush=True)
            elif res["status"] == "ok":
                pass  # has models but no :cloud ones
            if done % 200 == 0:
                print(f"  ...{done}/{len(ips)} probed, {len(results)} cloud-hits, {time.time()-t0:.0f}s", flush=True)

    # Build summary
    summary = {
        "total_scanned": len(ips),
        "hosts_with_cloud": len(results),
        "cloud_models": {}
    }
    for model, host_list in sorted(cloud_hosts.items(), key=lambda x: -len(x[1])):
        summary["cloud_models"][model] = {
            "count": len(host_list),
            "hosts": host_list
        }

    output = {"hosts": results, "cloud_summary": summary}
    json.dump(output, open(output_file, "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"TAG PROBE COMPLETE — {len(results)} hosts with :cloud models")
    print(f"{'='*60}")
    for model, info in sorted(summary["cloud_models"].items(), key=lambda x: -x[1]["count"]):
        print(f"  {model:40s} {info['count']:4d} hosts")
    print(f"{'='*60}")
    print(f"Full results: {output_file}")

if __name__ == "__main__":
    main()
