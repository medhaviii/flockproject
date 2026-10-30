#!/usr/bin/env python
"""
01_clean_raw_house1.py  --  RAW REFIT House 1: audit -> clean -> 1-minute resample -> plots.

Stages (all file reads are chunked):
  Pass 1  AUDIT      : streaming data-quality statistics on the untouched raw files.
  Pass 2  CLEAN      : sort/dedupe, outlier -> NaN, 1-min mean resample, short-gap fill, long-outage flags.
  Pass 3  PLOTS      : before (raw ~8 s samples) vs after (1-min cleaned) for Aggregate + 2 appliances.

Cleaning rules (all configurable by CLI flags, all reported in the outputs):
  * Canonical clock = Unix (UTC epoch seconds). The 'Time' string is only cross-checked against it.
  * Ordering: rows sorted by Unix (stable). If the raw files are already ordered this is a no-op.
  * Duplicates: rows sharing the same Unix second -> keep the FIRST occurrence, drop the rest.
  * Impossible values: negative power -> NaN; power > --cap-watts -> NaN (treated as sensor glitch).
  * Resample: arithmetic mean of all valid readings inside each UTC minute (NaNs ignored).
    Minutes with no valid reading stay NaN.
  * Short gaps: NaN runs of <= --max-gap-min minutes, bounded by valid values on both sides,
    are linearly interpolated (per channel). Everything longer is LEFT as NaN.
  * Long outages: minutes where Aggregate is still NaN after filling -> long_outage = True
    and listed in outages.csv. They are never imputed.

Outputs (in --out-dir):
  house1_1min_clean.csv.gz, audit_summary.json, audit_report.md, outages.csv, dt_histogram.csv,
  fig_before_after_typical.png, fig_before_after_outage.png (if a suitable outage exists),
  fig_before_after_custom.png (if --plot-start given)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHANNELS = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
TIME_FMT = "%Y-%m-%d %H:%M:%S"
LOCAL_TZ = "Europe/London"


# --------------------------------------------------------------------------- IO helpers
def read_chunks(paths, chunksize):
    """Yield (part_number, raw_chunk). Column names stripped; required columns checked."""
    required = ["Time", "Unix"] + CHANNELS
    for part, p in enumerate(paths, 1):
        first = True
        for chunk in pd.read_csv(p, chunksize=chunksize, dtype={"Time": str}, skipinitialspace=True):
            chunk.columns = chunk.columns.astype(str).str.strip()
            if first:
                missing = [c for c in required if c not in chunk.columns]
                if missing:
                    raise ValueError(f"{p}: missing columns {missing}. Found: {list(chunk.columns)}")
                first = False
            yield part, chunk


def coerce(chunk):
    """Numeric coercion. Returns (frame with Unix + channels, {col: count of non-numeric junk})."""
    out = pd.DataFrame({"Unix": pd.to_numeric(chunk["Unix"], errors="coerce")})
    junk = {"Unix": int((chunk["Unix"].notna() & out["Unix"].isna()).sum())}
    for c in CHANNELS:
        num = pd.to_numeric(chunk[c], errors="coerce")
        junk[c] = int((chunk[c].notna() & num.isna()).sum())
        out[c] = num.astype("float32")
    return out, junk


# --------------------------------------------------------------------------- PASS 1: audit
def run_audit(paths, chunksize, cap):
    A = {
        "files": [str(p) for p in paths],
        "rows_total": 0, "unix_missing": 0, "unix_nonnumeric": 0,
        "time_unparseable": 0, "tz_checked": 0,
        "time_mismatch_vs_london": 0, "time_mismatch_vs_utc": 0,
        "backward_steps": 0, "same_unix_adjacent": 0,
        "dup_exact_adjacent": 0, "dup_conflicting_adjacent": 0,
        "per_part": {}, "channels": {c: dict(nan=0, nonnumeric=0, negative=0, over_cap=0, zero=0,
                                               min=np.inf, max=-np.inf) for c in CHANNELS},
    }
    dt_small = np.zeros(31, dtype=np.int64)            # dt = 1..30 s
    dt_coarse = {"31-60s": 0, "61-300s": 0, "301-3600s": 0, ">3600s": 0}
    gaps = []                                           # (prev_unix, next_unix, seconds) for dt > 300 s
    carry_u, carry_v = None, None

    for part, ch in read_chunks(paths, chunksize):
        df, junk = coerce(ch)
        n = len(df)
        A["rows_total"] += n
        pp = A["per_part"].setdefault(part, dict(rows=0, first_unix=None, last_unix=None,
                                                  min_unix=np.inf, max_unix=-np.inf))
        pp["rows"] += n
        u = df["Unix"].to_numpy(dtype="float64")
        valid_u = u[~np.isnan(u)]
        if len(valid_u):
            if pp["first_unix"] is None:
                pp["first_unix"] = float(valid_u[0])
            pp["last_unix"] = float(valid_u[-1])
            pp["min_unix"] = min(pp["min_unix"], float(valid_u.min()))
            pp["max_unix"] = max(pp["max_unix"], float(valid_u.max()))
        A["unix_missing"] += int(np.isnan(u).sum())
        A["unix_nonnumeric"] += junk["Unix"]

        # --- Time string vs Unix consistency
        t = pd.to_datetime(ch["Time"], format=TIME_FMT, errors="coerce")
        A["time_unparseable"] += int(t.isna().sum())
        ok = df["Unix"].notna().to_numpy() & t.notna().to_numpy()
        if ok.any():
            uu = pd.to_datetime(df.loc[ok, "Unix"], unit="s", utc=True)
            local = uu.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
            utc = uu.dt.tz_localize(None)
            A["tz_checked"] += int(ok.sum())
            A["time_mismatch_vs_london"] += int(((local - t[ok]).abs().dt.total_seconds() > 2).sum())
            A["time_mismatch_vs_utc"] += int(((utc - t[ok]).abs().dt.total_seconds() > 2).sum())

        # --- per-channel value checks
        for c in CHANNELS:
            v = df[c].to_numpy(dtype="float64")
            s = A["channels"][c]
            s["nan"] += int(np.isnan(v).sum())
            s["nonnumeric"] += junk[c]
            with np.errstate(invalid="ignore"):
                s["negative"] += int((v < 0).sum())
                s["over_cap"] += int((v > cap).sum())
                s["zero"] += int((v == 0).sum())
            if not np.all(np.isnan(v)):
                s["min"] = min(s["min"], float(np.nanmin(v)))
                s["max"] = max(s["max"], float(np.nanmax(v)))

        # --- ordering, duplicates, sampling intervals (with carry-over from previous chunk)
        vals = df[CHANNELS].to_numpy(dtype="float64")
        if carry_u is not None:
            u_full = np.concatenate([[carry_u], u])
            v_full = np.vstack([carry_v, vals])
        else:
            u_full, v_full = u, vals
        d = np.diff(u_full)
        with np.errstate(invalid="ignore"):
            A["backward_steps"] += int((d < 0).sum())
            same = d == 0
            A["same_unix_adjacent"] += int(same.sum())
            if same.any():
                eq = ((v_full[1:] == v_full[:-1]) | (np.isnan(v_full[1:]) & np.isnan(v_full[:-1]))).all(axis=1)
                A["dup_exact_adjacent"] += int((same & eq).sum())
                A["dup_conflicting_adjacent"] += int((same & ~eq).sum())
            pos = d[d > 0]
            small = pos[pos <= 30].astype(int)
            dt_small += np.bincount(small, minlength=31)[:31]
            dt_coarse["31-60s"] += int(((pos > 30) & (pos <= 60)).sum())
            dt_coarse["61-300s"] += int(((pos > 60) & (pos <= 300)).sum())
            dt_coarse["301-3600s"] += int(((pos > 300) & (pos <= 3600)).sum())
            dt_coarse[">3600s"] += int((pos > 3600).sum())
            for i in np.where(d > 300)[0]:
                gaps.append((float(u_full[i]), float(u_full[i + 1]), float(d[i])))
        carry_u, carry_v = u[-1], vals[-1:].copy()
        print(f"  audit: part {part}, rows so far {A['rows_total']:,}", end="\r")
    print()

    # --- derived summaries
    pos_total = int(dt_small.sum() + sum(dt_coarse.values()))
    A["dt_positive_intervals"] = pos_total
    if pos_total:
        A["dt_mode_seconds"] = int(np.argmax(dt_small[1:]) + 1)
        A["dt_share_mode"] = float(dt_small[1:].max() / pos_total)
        cum = np.cumsum(dt_small) / pos_total
        idx = np.where(cum >= 0.5)[0]
        A["dt_median_seconds"] = int(idx[0]) if len(idx) else ">30"
    hist = {str(i): int(dt_small[i]) for i in range(1, 31)}
    hist.update(dt_coarse)
    A["dt_histogram"] = hist
    A["gaps_over_5min"] = {
        "count": len(gaps),
        "count_over_1h": int(sum(g[2] > 3600 for g in gaps)),
        "count_over_1day": int(sum(g[2] > 86400 for g in gaps)),
        "largest": sorted(gaps, key=lambda g: -g[2])[:10],
    }
    # part overlap check
    parts = sorted(A["per_part"])
    A["parts_overlap"] = False
    for a, b in zip(parts[:-1], parts[1:]):
        la, fb = A["per_part"][a]["max_unix"], A["per_part"][b]["min_unix"]
        if fb <= la:
            A["parts_overlap"] = True
    A["needs_global_sort"] = bool(A["backward_steps"] > 0 or A["parts_overlap"])
    for c in CHANNELS:  # make json-safe
        for k in ("min", "max"):
            if not np.isfinite(A["channels"][c][k]):
                A["channels"][c][k] = None
    return A


# --------------------------------------------------------------------------- PASS 2: clean
def fill_short_gaps(s: pd.Series, max_len: int):
    """Linearly fill NaN runs of length <= max_len that have valid values on both sides."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    short = isna & (run_len <= max_len)
    interp = s.interpolate(method="linear", limit_area="inside")
    filled = s.where(~short, interp)
    return filled, (short & filled.notna())


def iter_frames(paths, chunksize, global_sort):
    if global_sort:
        print("  NOTE: raw data not globally ordered -> in-memory stable sort (float32).")
        frames = [coerce(ch)[0] for _, ch in read_chunks(paths, chunksize)]
        df = pd.concat(frames, ignore_index=True).sort_values("Unix", kind="stable")
        yield df
    else:
        for _, ch in read_chunks(paths, chunksize):
            yield coerce(ch)[0]


def run_clean(paths, chunksize, cap, max_gap, global_sort):
    stats = dict(rows_read=0, rows_unix_nan_dropped=0, duplicate_rows_dropped=0,
                 negative_to_nan={c: 0 for c in CHANNELS}, over_cap_to_nan={c: 0 for c in CHANNELS})
    acc_sum = acc_cnt = acc_n = None
    carry = -np.inf

    for df in iter_frames(paths, chunksize, global_sort):
        stats["rows_read"] += len(df)
        n0 = len(df)
        df = df[df["Unix"].notna()]
        stats["rows_unix_nan_dropped"] += n0 - len(df)
        n1 = len(df)
        if not global_sort:
            df = df.sort_values("Unix", kind="stable")
        df = df.drop_duplicates(subset="Unix", keep="first")
        df = df[df["Unix"] > carry]                     # drops duplicates straddling chunk boundaries
        stats["duplicate_rows_dropped"] += n1 - len(df)
        if len(df) == 0:
            continue
        carry = float(df["Unix"].iloc[-1])

        for c in CHANNELS:
            v = df[c].astype("float64")
            neg, big = v < 0, v > cap
            stats["negative_to_nan"][c] += int(neg.sum())
            stats["over_cap_to_nan"][c] += int(big.sum())
            df[c] = v.mask(neg | big)

        minute = (df["Unix"].astype("int64") // 60).rename("minute")
        g = df[CHANNELS].astype("float64").groupby(minute)
        s, cn, nn = g.sum(), g.count(), minute.groupby(minute).size().rename("n_readings")
        acc_sum = s if acc_sum is None else acc_sum.add(s, fill_value=0)
        acc_cnt = cn if acc_cnt is None else acc_cnt.add(cn, fill_value=0)
        acc_n = nn if acc_n is None else acc_n.add(nn, fill_value=0)
        print(f"  clean: processed up to Unix {carry:.0f}", end="\r")
    print()

    full = np.arange(acc_sum.index.min(), acc_sum.index.max() + 1)
    S = acc_sum.reindex(full).fillna(0.0)
    C = acc_cnt.reindex(full).fillna(0.0)
    N = acc_n.reindex(full).fillna(0).astype("int32")
    clean = S.where(C > 0) / C.where(C > 0)
    idx = pd.to_datetime(full * 60, unit="s", utc=True)
    clean.index = idx
    clean.index.name = "timestamp_utc"
    N.index = idx

    stats["minutes_in_span"] = int(len(clean))
    stats["minutes_with_any_reading"] = int((N > 0).sum())
    miss_before, filled_mask, filled_n, remaining = {}, {}, {}, {}
    for c in CHANNELS:
        miss_before[c] = int(clean[c].isna().sum())
        clean[c], filled_mask[c] = fill_short_gaps(clean[c], max_gap)
        filled_n[c] = int(filled_mask[c].sum())
        remaining[c] = int(clean[c].isna().sum())
    fm = pd.DataFrame(filled_mask)
    stats["minutes_missing_before_fill"] = miss_before
    stats["minutes_filled"] = filled_n
    stats["minutes_missing_after_fill"] = remaining

    out = clean.copy()
    out["n_readings"] = N
    out["n_filled"] = fm.sum(axis=1).astype("int8")
    out["long_outage"] = out["Aggregate"].isna()
    out.insert(0, "timestamp_local", out.index.tz_convert(LOCAL_TZ).tz_localize(None))
    return out, fm, stats


def list_outages(clean):
    isn = clean["long_outage"].to_numpy()
    if not isn.any():
        return pd.DataFrame(columns=["start_utc", "end_utc", "minutes", "hours"])
    edge = np.diff(np.concatenate([[0], isn.astype(int), [0]]))
    starts, ends = np.where(edge == 1)[0], np.where(edge == -1)[0] - 1
    idx = clean.index
    df = pd.DataFrame({"start_utc": idx[starts], "end_utc": idx[ends], "minutes": ends - starts + 1})
    df["hours"] = (df["minutes"] / 60).round(2)
    return df.sort_values("minutes", ascending=False).reset_index(drop=True)


def flatline_report(agg, min_len=120):
    grp = (agg != agg.shift()).cumsum()
    size = agg.groupby(grp).transform("size")
    flat = (size >= min_len) & agg.notna()
    return dict(min_run_minutes=min_len, runs=int(grp[flat].nunique()),
                longest_run_minutes=int(size[flat].max()) if flat.any() else 0)


# --------------------------------------------------------------------------- PASS 3: plots
def read_windows(paths, chunksize, windows, cols):
    store = [[] for _ in windows]
    for _, ch in read_chunks(paths, chunksize):
        u = pd.to_numeric(ch["Unix"], errors="coerce")
        for k, (t0, t1) in enumerate(windows):
            m = (u >= t0) & (u < t1)
            if m.any():
                d = pd.DataFrame({"Unix": u[m]})
                for c in cols:
                    d[c] = pd.to_numeric(ch.loc[m, c], errors="coerce")
                store[k].append(d)
    return [pd.concat(s).sort_values("Unix", kind="stable") if s else pd.DataFrame(columns=["Unix"] + cols)
            for s in store]


def plot_window(raw, clean, fm, t0, t1, chans, title, path):
    fig, axes = plt.subplots(len(chans), 2, figsize=(15, 3.1 * len(chans)), sharex="col")
    axes = np.atleast_2d(axes)
    cl = clean.loc[t0:t1]
    x_raw = pd.to_datetime(raw["Unix"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    x_cl = cl.index.tz_convert(LOCAL_TZ)
    for r, c in enumerate(chans):
        a0, a1 = axes[r]
        a0.plot(x_raw, raw[c], lw=0.5, color="0.35")
        a0.set_ylabel(f"{c} (W)")
        a1.plot(x_cl, cl[c], lw=0.8, color="tab:blue")
        f = fm[c].loc[t0:t1].to_numpy()
        if f.any():
            a1.plot(x_cl[f], cl[c].to_numpy()[f], ".", color="orange", ms=4, label="gap-filled")
            a1.legend(loc="upper right", fontsize=8)
        lo = cl["long_outage"].to_numpy()
        if lo.any():
            a1.fill_between(x_cl, 0, 1, where=lo, transform=a1.get_xaxis_transform(),
                            color="red", alpha=0.15, label="long outage (not imputed)")
        if r == 0:
            a0.set_title("BEFORE: raw samples")
            a1.set_title("AFTER: 1-min cleaned (red = long outage, orange = filled)")
    for a in axes[-1]:
        a.set_xlabel("local time (Europe/London)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def pick_typical_start(clean):
    cov = clean["Aggregate"].notna().groupby(clean.index.floor("D")).mean()
    v = cov.to_numpy()
    cand = np.where((v[:-1] >= 0.99) & (v[1:] >= 0.99))[0]
    i = cand[np.argmin(np.abs(cand - len(v) / 2))] if len(cand) else int(np.argmax(v[:-1] + v[1:]))
    return cov.index[i]


# --------------------------------------------------------------------------- report
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def ts(u):
    return pd.to_datetime(u, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")


def write_report(A, S, outages, flat, args, path):
    L = ["# Raw House 1 audit and cleaning report (auto-generated)", ""]
    L += ["## 1. Files and coverage", ""]
    L.append(md_table(["Part", "Rows", "First Unix (file order)", "Last Unix (file order)"],
                      [[p, f"{d['rows']:,}", ts(d["first_unix"]), ts(d["last_unix"])]
                       for p, d in A["per_part"].items()]))
    L += ["", f"Total raw rows: **{A['rows_total']:,}**. Parts overlap in time: **{A['parts_overlap']}**.", ""]
    L += ["## 2. Timestamps, ordering, duplicates", ""]
    L.append(md_table(["Check", "Count"], [
        ["Unix missing", f"{A['unix_missing']:,}"], ["Unix non-numeric", f"{A['unix_nonnumeric']:,}"],
        ["'Time' string unparseable", f"{A['time_unparseable']:,}"],
        ["Rows cross-checked Time vs Unix", f"{A['tz_checked']:,}"],
        ["Time != Unix as Europe/London (>2 s)", f"{A['time_mismatch_vs_london']:,}"],
        ["Time != Unix as UTC (>2 s)", f"{A['time_mismatch_vs_utc']:,}"],
        ["Backward time steps (out of order)", f"{A['backward_steps']:,}"],
        ["Adjacent rows with identical Unix", f"{A['same_unix_adjacent']:,}"],
        ["  of which exact duplicates", f"{A['dup_exact_adjacent']:,}"],
        ["  of which conflicting values", f"{A['dup_conflicting_adjacent']:,}"],
    ]))
    L += ["", "## 3. Sampling intervals", ""]
    L.append(f"Positive intervals: {A['dt_positive_intervals']:,}; mode = {A.get('dt_mode_seconds')} s "
             f"({A.get('dt_share_mode', 0):.1%} of intervals); median = {A.get('dt_median_seconds')} s.")
    g = A["gaps_over_5min"]
    L += ["", f"Gaps > 5 min: {g['count']:,} (> 1 h: {g['count_over_1h']:,}; > 1 day: {g['count_over_1day']:,}). "
              "Ten largest:", ""]
    L.append(md_table(["From", "To", "Hours"], [[ts(a), ts(b), f"{s/3600:.2f}"] for a, b, s in g["largest"]]))
    L += ["", "Full interval histogram: `dt_histogram.csv`.", ""]
    L += ["## 4. Per-channel raw value checks", ""]
    rows = []
    for c in CHANNELS:
        s = A["channels"][c]
        rows.append([c, f"{s['nan']:,}", f"{s['nonnumeric']:,}", f"{s['negative']:,}",
                     f"{s['over_cap']:,}", f"{s['zero']:,}", s["min"], s["max"]])
    L.append(md_table(["Channel", "NaN", "Non-numeric", "Negative", f"> {args.cap_watts:g} W", "Zero",
                       "Min", "Max"], rows))
    L += ["", f"## 5. Cleaning actions (gap-fill limit = {args.max_gap_min} min, cap = {args.cap_watts:g} W)", ""]
    L.append(f"Global sort needed: **{A['needs_global_sort']}**. Duplicate-Unix rows dropped (keep first): "
             f"**{S['duplicate_rows_dropped']:,}**. Rows with missing Unix dropped: {S['rows_unix_nan_dropped']:,}.")
    L += ["", f"1-min span: {S['minutes_in_span']:,} minutes; minutes with >=1 reading: "
              f"{S['minutes_with_any_reading']:,}.", ""]
    L.append(md_table(["Channel", "Neg->NaN", "Over cap->NaN", "Missing min (pre-fill)", "Filled", "Missing (post-fill)"],
                      [[c, S["negative_to_nan"][c], S["over_cap_to_nan"][c], S["minutes_missing_before_fill"][c],
                        S["minutes_filled"][c], S["minutes_missing_after_fill"][c]] for c in CHANNELS]))
    L += ["", "## 6. Long outages (Aggregate still missing after fill; never imputed)", ""]
    if len(outages):
        L.append(f"Count: **{len(outages)}**; total missing: **{int(outages['minutes'].sum()):,} min**.")
        L += ["", "Longest 15:", ""]
        L.append(md_table(["Start UTC", "End UTC", "Minutes", "Hours"],
                          [[r.start_utc, r.end_utc, r.minutes, r.hours] for r in outages.head(15).itertuples()]))
    else:
        L.append("None.")
    L += ["", f"## 7. Flat-line check (Aggregate)", "",
          f"Runs of >= {flat['min_run_minutes']} identical consecutive 1-min values: **{flat['runs']}**; "
          f"longest = {flat['longest_run_minutes']} min. (Reported only, not removed.)", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-files", nargs="+", default=[r"C:\medhavi\proj\refit\RAW_House1_Part1.csv",
                                                       r"C:\medhavi\proj\refit\RAW_House1_Part2.csv"])
    ap.add_argument("--out-dir", default=r"C:\medhavi\proj\outputs\raw_house1")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--max-gap-min", type=int, default=5, help="fill NaN runs up to this many minutes")
    ap.add_argument("--cap-watts", type=float, default=25_000.0, help="values above this are impossible -> NaN")
    ap.add_argument("--plot-appliances", nargs=2, default=None, help="e.g. Appliance1 Appliance5")
    ap.add_argument("--plot-start", default=None, help="optional extra 48h window, e.g. 2014-03-01 (UTC)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.raw_files]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("[1/3] Auditing raw files ...")
    A = run_audit(paths, args.chunksize, args.cap_watts)
    pd.DataFrame({"interval": list(A["dt_histogram"]), "count": list(A["dt_histogram"].values())}) \
        .to_csv(out / "dt_histogram.csv", index=False)

    print("[2/3] Cleaning and resampling to 1 minute ...")
    clean, fm, S = run_clean(paths, args.chunksize, args.cap_watts, args.max_gap_min, A["needs_global_sort"])
    outages = list_outages(clean)
    flat = flatline_report(clean["Aggregate"])
    outages.to_csv(out / "outages.csv", index=False)
    cols_out = ["timestamp_local"] + CHANNELS + ["n_readings", "n_filled", "long_outage"]
    clean[cols_out].round(2).to_csv(out / "house1_1min_clean.csv.gz", compression="gzip")
    summary = dict(args=vars(args), audit=A, cleaning=S, outages=dict(count=len(outages),
                   total_minutes=int(outages["minutes"].sum()) if len(outages) else 0), flatline=flat)
    (out / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_report(A, S, outages, flat, args, out / "audit_report.md")

    print("[3/3] Plotting before/after ...")
    if args.plot_appliances:
        apps = list(args.plot_appliances)
    else:
        means = clean[[f"Appliance{i}" for i in range(1, 10)]].mean().sort_values(ascending=False)
        apps = list(means.index[:2])
        print(f"  auto-selected appliances (highest mean power): {apps}")
    chans = ["Aggregate"] + apps
    day = pd.Timedelta(days=1)
    wins = [("typical", "Typical well-covered 48 h", pick_typical_start(clean), pick_typical_start(clean) + 2 * day)]
    short = outages[outages["minutes"] <= 1440] if len(outages) else outages
    if len(short):
        o = short.iloc[0]
        wins.append(("outage", f"48 h-ish window around a {o.minutes}-min outage",
                     o.start_utc - pd.Timedelta(hours=12), o.end_utc + pd.Timedelta(hours=12)))
    else:
        print("  no outage <= 24 h found; skipping outage figure (use --plot-start for a custom window).")
    if args.plot_start:
        s0 = pd.Timestamp(args.plot_start, tz="UTC")
        wins.append(("custom", f"Custom window from {args.plot_start}", s0, s0 + 2 * day))
    raws = read_windows(paths, args.chunksize, [(int(w[2].timestamp()), int(w[3].timestamp())) for w in wins], chans)
    for (tag, title, t0, t1), raw in zip(wins, raws):
        plot_window(raw, clean, fm, t0, t1, chans, f"RAW House 1 - {title}", out / f"fig_before_after_{tag}.png")

    print(f"\nDone in {time.time() - t_start:.0f}s. Outputs in: {out}")
    print("Next: open audit_report.md and paste it back so DATA.md can be written from the real numbers.")


if __name__ == "__main__":
    main()
