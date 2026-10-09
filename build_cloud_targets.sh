#!/bin/bash
# build_cloud_targets.sh — fetch cloud provider CIDR ranges for masscan targeting
# Output: cloud_targets.txt (one CIDR per line)

set -euo pipefail
OUT="cloud_targets.txt"

echo "[+] Fetching cloud provider CIDR ranges..."

# --- AWS ---
echo "[+] AWS..."
curl -s "https://ip-ranges.amazonaws.com/ip-ranges.json" \
  | python3 -c "import sys,json; data=json.load(sys.stdin); [print(r['ip_prefix']) for r in data['prefixes']]" \
  > /tmp/aws_ranges.txt 2>/dev/null || echo "  [-] AWS failed"

# --- GCP ---
echo "[+] GCP..."
curl -s "https://www.gstatic.com/ipranges/cloud.json" \
  | python3 -c "import sys,json; data=json.load(sys.stdin); [print(r['ipv4Prefix']) for r in data['prefixes'] if 'ipv4Prefix' in r]" \
  > /tmp/gcp_ranges.txt 2>/dev/null || echo "  [-] GCP failed"

# --- Oracle Cloud ---
echo "[+] Oracle Cloud..."
curl -s "https://docs.oracle.com/iaas/tools/public_ip_ranges.json" \
  | python3 -c "import sys,json; data=json.load(sys.stdin); [print(c['cidr']) for r in data['regions'] for c in r.get('cidrs',[])]" \
  > /tmp/oracle_ranges.txt 2>/dev/null || echo "  [-] Oracle failed"

# --- Static fallback ranges (Hetzner, Contabo, OVH, Linode, Vultr) ---
echo "[+] Static ranges..."
cat > /tmp/static_ranges.txt << 'STATICEOF'
# Hetzner
65.108.0.0/16
65.109.0.0/16
78.46.0.0/15
116.202.0.0/15
135.181.0.0/16
136.243.0.0/16
138.201.0.0/16
142.132.0.0/16
144.76.0.0/16
148.251.0.0/16
159.69.0.0/16
162.55.0.0/16
176.9.0.0/16
178.63.0.0/16
188.40.0.0/16
195.201.0.0/16
5.9.0.0/16
46.4.0.0/16
# Vultr
45.32.0.0/16
45.63.0.0/16
45.76.0.0/16
45.77.0.0/16
64.227.0.0/16
95.179.0.0/16
108.61.0.0/16
140.82.0.0/17
142.93.0.0/16
144.202.0.0/16
149.28.0.0/16
159.203.0.0/16
167.71.0.0/16
167.99.0.0/16
174.138.0.0/16
STATICEOF

# --- Combine and deduplicate ---
echo "[+] Combining and deduplicating..."
cat /tmp/aws_ranges.txt /tmp/gcp_ranges.txt /tmp/oracle_ranges.txt \
    /tmp/static_ranges.txt 2>/dev/null \
  | grep -vE '^\s*$|^\s*#' \
  | sort -u > "$OUT"

TOTAL_RANGES=$(wc -l < "$OUT")
echo "[+] Total unique CIDR ranges: $TOTAL_RANGES"

python3 -c "
import ipaddress
total = 0
with open('$OUT') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            net = ipaddress.ip_network(line, strict=False)
            total += net.num_addresses
        except ValueError:
            pass
print(f'[+] Total IPv4 addresses in scope: {total:,}')
print(f'    At 10,000 pps (GitHub Actions rate): ~{total/10000/60:.1f} minutes')
"
