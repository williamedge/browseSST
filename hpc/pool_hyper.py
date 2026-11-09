#!/usr/bin/env python
"""Step 3b: pool the per-tile hyperparameter fits (median in log space) into the shared hyper.json that every tile job uses; prints the spread between tiles per parameter.

  python hpc/pool_hyper.py --inputs hyper/hyper_*.json --out hyper.json
"""
import argparse

import _util


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from inversion_sst_gp.spacetime import pipeline as P
    hs = [P.Hyper.load(p) for p in a.inputs]
    pooled = P.pool_hypers(hs)
    pooled.save(a.out)
    _util.log(f"pooled {len(hs)} fits -> {a.out}")
    for k, v in pooled.meta["spread_max_over_min"].items():
        val = pooled.stage1.get(k, pooled.stage2.get(k))
        _util.log(f"  {k:12s} {val:11.4g}   max/min over tiles {v:6.2f}")


if __name__ == "__main__":
    main()
