"""SMOKE-TEST ONLY: generate a small synthetic dataset in REFIT file format (NOT real data).

Usage:  set REFIT_PROJECT_ROOT=C:\\temp\\synth   (a scratch folder, NOT your real project)
        python tools\\make_synthetic_refit.py --days 60
It writes <root>\\refit\\RAW_House1_Part1/2.csv, <root>\\clean\\CLEAN_House{1,2,3,5}.csv and a SYNTHETIC.flag file
so that the PDF builder stamps a 'SYNTHETIC' banner on any report produced from it.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("REFIT_PROJECT_ROOT", "synthetic_root"))
WM = {1: 5, 2: 2, 3: 6, 5: 3}


def sig_events(n, rng, per_day, dur, power, days, hours=None):
    x = np.zeros(n)
    for d in range(days):
        for _ in range(rng.poisson(per_day)):
            h = rng.uniform(6, 22) if hours is None else rng.choice(hours) + rng.uniform(-1, 1)
            s = int((d * 86400 + h * 3600) / 8)
            L = int(dur * 60 / 8)
            if s + L < n:
                x[s:s + L] += power * (1 + 0.1 * rng.standard_normal(L))
    return x


def wm_profile(rng):
    # fill/wash 220 W 15 min, heat 2000 W 18 min, wash 250 W 20 min, short pause, spin 450 W 6 min
    segs = [(15, 220), (18, 2000), (20, 250), (3, 5), (6, 450)]
    return np.concatenate([np.full(int(m * 60 / 8), p) * (1 + 0.05 * rng.standard_normal(int(m * 60 / 8))) for m, p in segs])


def make_house(h, days, seed, start="2014-01-06"):
    rng = np.random.default_rng(seed)
    n = days * 86400 // 8
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    unix = t0 + np.arange(n) * 8
    hour = (unix % 86400) / 3600.0
    sig = {}
    sig["fridge"] = 90 * (((unix // 60) % 40) < 14)
    sig["freezer"] = 70 * (((unix // 60) % 55) < 20)
    sig["kettle"] = sig_events(n, rng, 4, 3, 2200, days)
    sig["tv"] = sig_events(n, rng, 1.2, 150, 110, days, hours=[19])
    sig["dishwasher"] = sig_events(n, rng, 0.5, 60, 1200, days, hours=[20])
    sig["micro"] = sig_events(n, rng, 1.5, 4, 1100, days)
    sig["pc"] = sig_events(n, rng, 0.8, 240, 90, days, hours=[10, 15])
    sig["toaster"] = sig_events(n, rng, 1.0, 3, 900, days, hours=[8])
    sig["misc"] = np.abs(rng.standard_normal(n)) * 3
    wm = np.zeros(n)
    prof = wm_profile(rng)
    wm_day = 0.45 if h != 3 else 0.25
    for d in range(days):
        if rng.random() < wm_day:
            hh = rng.choice([9, 18]) + rng.uniform(-1.5, 1.5)
            s = int((d * 86400 + hh * 3600) / 8)
            if s + len(prof) < n:
                wm[s:s + len(prof)] += prof
    others = [k for k in sig if k != "misc"][:8]
    app = {}
    ai = 1
    for c in range(1, 10):
        if c == WM[h]:
            app[f"Appliance{c}"] = wm
        else:
            k = others[(ai - 1) % len(others)] if ai <= 8 else "misc"
            app[f"Appliance{c}"] = sig[k]
            ai += 1
    base = 120 + 40 * ((hour > 17) & (hour < 23)) + 15 * rng.standard_normal(n)
    agg = base + sum(v for v in app.values()) * (1.0)
    return unix, agg, app


def fmt_time(unix):
    return pd.to_datetime(unix, unit="s", utc=True).tz_convert("Europe/London").tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")


def frame(unix, agg, app):
    df = pd.DataFrame({"Time": fmt_time(unix), "Unix": unix.astype("int64"), "Aggregate": np.round(agg, 1)})
    for k, v in app.items():
        df[k] = np.round(v, 1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    a = ap.parse_args()
    (ROOT / "refit").mkdir(parents=True, exist_ok=True)
    (ROOT / "clean").mkdir(parents=True, exist_ok=True)
    (ROOT / "SYNTHETIC.flag").write_text("synthetic smoke-test data", encoding="utf-8")
    rng = np.random.default_rng(99)

    # ---- RAW House 1: irregular sampling, duplicates (exact + conflicting), NaNs, outages, spikes
    unix, agg, app = make_house(1, a.days, 1)
    df = frame(unix, agg, app)
    keep = rng.random(len(df)) > 0.04                              # irregular intervals
    df = df[keep].reset_index(drop=True)
    cut = (df["Unix"] > df["Unix"].iloc[0] + 20 * 86400) & (df["Unix"] < df["Unix"].iloc[0] + 20 * 86400 + 3 * 3600)
    df = df[~cut]                                                   # 3 h outage
    cut = (df["Unix"] > df["Unix"].iloc[0] + 35 * 86400) & (df["Unix"] < df["Unix"].iloc[0] + 36.2 * 86400)
    df = df[~cut].reset_index(drop=True)                            # ~29 h outage
    short = (df["Unix"] > df["Unix"].iloc[0] + 10 * 86400) & (df["Unix"] < df["Unix"].iloc[0] + 10 * 86400 + 150)
    df = df[~short].reset_index(drop=True)                          # short gap
    ex = df.sample(300, random_state=1)
    cf = df.sample(150, random_state=2).copy()
    cf["Aggregate"] = cf["Aggregate"] + rng.normal(0, 25, len(cf)).round(1)
    df = pd.concat([df, ex, cf]).sort_index(kind="stable").reset_index(drop=True)
    sp = rng.choice(len(df), 6, replace=False)
    df.loc[sp[:3], "Aggregate"] = 60000.0                            # impossible spikes
    df.loc[sp[3:], "Aggregate"] = -5.0
    split = int(len(df) * 0.55)
    p1, p2 = df.iloc[:split].copy(), df.iloc[split:].copy()
    miss = rng.random(len(p2)) < 0.004                               # missing values in Part 2
    p2.loc[miss, [c for c in p2.columns if c.startswith("Appliance") or c == "Aggregate"]] = np.nan
    p1.to_csv(ROOT / "refit" / "RAW_House1_Part1.csv", index=False)
    p2.to_csv(ROOT / "refit" / "RAW_House1_Part2.csv", index=False)

    # ---- CLEAN houses (8 s grid, Issues flag, a few NaN stretches)
    for h, seed, days in [(1, 11, a.days + 15), (2, 12, a.days + 5), (3, 13, a.days + 25), (5, 14, a.days)]:
        unix, agg, app = make_house(h, days, seed)
        df = frame(unix, agg, app)
        df["Issues"] = 0
        g0 = len(df) // 3
        df.loc[g0:g0 + 900, [c for c in df.columns if c.startswith("Appliance") or c == "Aggregate"]] = np.nan
        df.loc[g0:g0 + 900, "Issues"] = 1
        df = df.drop(df.index[len(df) // 2: len(df) // 2 + 2000])
        df.to_csv(ROOT / "clean" / f"CLEAN_House{h}.csv", index=False)
    print("synthetic data written to", ROOT)


if __name__ == "__main__":
    main()
