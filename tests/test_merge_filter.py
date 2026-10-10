"""Regression tests for the wave merge/filter stage.

The wave-3 filter kept hosts with >=10 open ports (firewalls) and discarded the
real servers. These tests pin the corrected direction so it cannot regress.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import merge_filter as mf


def test_middlebox_with_10_ports_is_dropped():
    ip_ports = {"1.1.1.1": set(range(1, 11))}  # 10 ports open -> firewall
    candidates, middleboxes = mf.split_candidates(ip_ports)
    assert "1.1.1.1" not in candidates, "10-port host must NOT be a candidate"
    assert "1.1.1.1" in middleboxes


def test_gateway_with_few_ports_is_kept():
    ip_ports = {"2.2.2.2": {8080, 3000, 8000}}  # wave-3 gateway shape
    candidates, middleboxes = mf.split_candidates(ip_ports)
    assert "2.2.2.2" in candidates
    assert "2.2.2.2" not in middleboxes


def test_threshold_boundary_is_exclusive_of_candidates():
    ip_ports = {
        "3.3.3.3": set(range(9)),           # 9 ports -> candidate (<10)
        "4.4.4.4": set(range(10)),          # 10 ports -> middlebox
    }
    candidates, middleboxes = mf.split_candidates(ip_ports)
    assert set(candidates) == {"3.3.3.3"}
    assert set(middleboxes) == {"4.4.4.4"}


def test_inverted_wave3_predicate_would_fail_this_assertion():
    """Guard: the old `len(ps) >= MIN_PORTS` kept firewalls, not servers."""
    ip_ports = {"5.5.5.5": set(range(12)), "6.6.6.6": {8080}}
    # corrected behaviour
    candidates, _ = mf.split_candidates(ip_ports)
    assert candidates == {"6.6.6.6": {8080}}
    # the old (buggy) direction would have produced the opposite
    old_kept = {ip for ip, ps in ip_ports.items() if len(ps) >= 10}
    assert old_kept == {"5.5.5.5"}
    assert old_kept != set(candidates)


def test_pick_probe_port_prefers_gateway_port():
    assert mf.pick_probe_port({9200, 8080, 5000}) == 8080
    assert mf.pick_probe_port({3000, 8000}) == 3000
    assert mf.pick_probe_port({2375}) == 2375


def test_load_hits_tolerates_trailing_comma(tmp_path):
    p = tmp_path / "hits_0.json"
    p.write_text('[{"ip": "7.7.7.7", "ports": [{"port": 8080, "status": "open"}]},\n')
    recs = mf.load_hits(str(p))
    assert recs and recs[0]["ip"] == "7.7.7.7"


def test_end_to_end_keeps_only_low_port_hosts(tmp_path):
    hits = tmp_path / "hits"
    (hits / "shard_0").mkdir(parents=True)
    (hits / "shard_0" / "hits_0.json").write_text(json.dumps([
        {"ip": "8.8.8.8", "ports": [{"port": p, "status": "open"} for p in range(1, 16)]},
        {"ip": "9.9.9.9", "ports": [{"port": 8080, "status": "open"},
                                    {"port": 3000, "status": "open"}]},
    ]))
    out = tmp_path / "out"
    os.environ["HITS_DIR"] = str(hits)
    os.environ["OUT_DIR"] = str(out)
    importlib.reload(mf)
    assert mf.main() == 0

    pass1 = (out / "hosts_pass1.txt").read_text().splitlines()
    assert pass1 == ["9.9.9.9:8080"], pass1  # firewall excluded, best port chosen

    clean = [json.loads(l) for l in (out / "candidates_clean.jsonl").read_text().splitlines()]
    assert [c["ip"] for c in clean] == ["9.9.9.9"]

    mid = [json.loads(l) for l in (out / "middleboxes.jsonl").read_text().splitlines()]
    assert [c["ip"] for c in mid] == ["8.8.8.8"]
