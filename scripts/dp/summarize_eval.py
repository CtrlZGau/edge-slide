"""Merge sharded eval results (<out>/<shard>/<mode>/episodes.jsonl) into per-mode success rates.
Prints a table and writes <out>/summary.json. Wilson 95% intervals for n=50 are wide (~±10 pts)."""
import json, math, sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
rows = defaultdict(dict)
for f in out.glob("*/*/episodes.jsonl"):
    mode = f.parent.name
    for l in f.read_text().splitlines():
        r = json.loads(l); rows[mode][r["seed"]] = r

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

summ = {}
def add(name, oks):
    k, n = sum(oks), len(oks); lo, hi = wilson(k, n)
    summ[name] = dict(success=k, n=n, rate=k / max(1, n), ci95=[round(lo, 3), round(hi, 3)])
    print(f"{name:28s} {k:3d}/{n:<3d} = {100 * k / max(1, n):5.1f}%   95% CI [{100 * lo:4.1f}, {100 * hi:4.1f}]")

if "cube" in rows: add("cube pick (C)", [r["success"] for r in rows["cube"].values()])
if "chain" in rows:
    ch = list(rows["chain"].values())
    add("push (A, chain stage 1)", [r["a_success"] for r in ch])
    add("pick after DP push (B|A)", [r["success"] for r in ch if r["a_success"]])
    add("push->pick chained (A->B)", [r["success"] for r in ch])
    for e, sgn in (("+y", 1), ("-y", -1)):
        sub = [r for r in ch if r.get("a_success") and r.get("plate_y_after_a") is not None and sgn * r["plate_y_after_a"] > 0]
        if sub: add(f"  chain, pushed to {e} edge", [r["success"] for r in sub])
if "pick" in rows:
    pk = [r for r in rows["pick"].values() if not r.get("excluded")]
    add("pick standalone (B)", [r["success"] for r in pk])
    for e in ("+y", "-y"):
        sub = [r for r in pk if r.get("edge") == e]
        if sub: add(f"  pick, {e} edge", [r["success"] for r in sub])
(out / "summary.json").write_text(json.dumps(summ, indent=2))
