#!/usr/bin/env python
"""
02_prepare_clean_1min.py -- OFFICIAL CLEANED REFIT: chunked 1-minute resampling + washing-machine mapping check.

For each house in the claimed washing-machine map (default Houses 1, 2, 3, 5) this script
  1. streams CLEAN_House<N>.csv in chunks and resamples ALL channels to 1-minute means (no gap filling;
     missing minutes stay NaN, negatives/>25 kW -> NaN and are counted),
  2. computes a washing-machine *signature* score (0-4) for every appliance channel,
  3. writes outputs/clean_1min/House<N>_1min.csv.gz, meta.json, mapping_verification.csv/json, wm_mapping.json.

IMPORTANT: the signature check is a plausibility heuristic (low duty cycle, 20-180 min cycles, several
cycles/week, high-power element). It cannot tell a washing machine from a dishwasher or tumble dryer. The
authoritative source for channel names is the REFIT readme; status 'consistent' only means the claimed
channel *looks like* a wet appliance.

Usage: python src\\02_prepare_clean_1min.py [--extra-map 4=Appliance4] [--drop-failed] [--force]
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from refit_common import (APPLIANCES, CLEAN_DIR, OUT_1MIN, WM_MAP_CLAIMED, jdump, signature, stream_minute_means)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra-map", nargs="*", default=[], help="extra houses, e.g. 4=Appliance4")
    ap.add_argument("--drop-failed", action="store_true", help="drop houses whose claimed channel scores < 3")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--force", action="store_true", help="recompute even if the 1-min file exists")
    args = ap.parse_args()

    mapping = dict(WM_MAP_CLAIMED)
    for e in args.extra_map:
        h, ch = e.split("=")
        mapping[int(h)] = ch
    OUT_1MIN.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_1MIN / "meta.json"
    meta = {}
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    sig_rows, wm_map = [], {}
    for h, claimed in mapping.items():
        src = CLEAN_DIR / f"CLEAN_House{h}.csv"
        dst = OUT_1MIN / f"House{h}_1min.csv.gz"
        if not src.exists():
            print(f"[House {h}] {src} not found - skipped")
            continue
        t0 = time.time()
        if dst.exists() and not args.force and str(h) in meta:
            print(f"[House {h}] reusing {dst.name}")
            df = pd.read_csv(dst, index_col=0)
        else:
            print(f"[House {h}] streaming {src.name} ...")
            df, m = stream_minute_means(src, args.chunksize)
            df.to_csv(dst, compression="gzip", float_format="%.2f")
            meta[str(h)] = m
            jdump(meta, meta_path)
            print(f"[House {h}] {m['minutes_in_span']:,} min, tz convention '{m['time_convention']}', "
                  f"neg->NaN {m['n_negative']:,}, over-cap->NaN {m['n_over_cap']:,}  ({time.time() - t0:.0f}s)")
        sigs = {c: signature(df[c].to_numpy(dtype="float64")) for c in APPLIANCES if c in df.columns}
        for c, s in sigs.items():
            sig_rows.append(dict(house=h, channel=c, claimed=(c == claimed), **s))
        cs = sigs.get(claimed, dict(score=0))
        similar = [c for c, s in sigs.items() if c != claimed and s["score"] >= 3]
        wm_map[str(h)] = dict(channel=claimed, signature_score=cs["score"],
                              status="consistent" if cs["score"] >= 3 else "check",
                              other_channels_with_similar_signature=similar)

    if not wm_map:
        raise SystemExit("No CLEAN files found. Check CLEAN_DIR (default <project>\\clean).")
    pd.DataFrame(sig_rows).to_csv(OUT_1MIN / "mapping_verification.csv", index=False)
    if args.drop_failed:
        wm_map = {h: v for h, v in wm_map.items() if v["status"] == "consistent"}
    jdump(wm_map, OUT_1MIN / "wm_mapping.json")
    jdump(wm_map, OUT_1MIN / "mapping_verification.json")
    print("\nWashing-machine channel signature check (0-4; >=3 = consistent):")
    for h, v in wm_map.items():
        print(f"  House {h}: {v['channel']}  score={v['signature_score']}  {v['status']}  "
              f"other similar channels: {v['other_channels_with_similar_signature']}")
    if any(v["status"] == "check" for v in wm_map.values()):
        print("\n  WARNING: at least one claimed channel did not look like a washing machine. Confirm against the "
              "REFIT readme before trusting results for that house (or rerun with --drop-failed).")


if __name__ == "__main__":
    main()
