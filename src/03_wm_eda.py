#!/usr/bin/env python
"""
03_wm_eda.py -- washing-machine exploratory analysis on the OFFICIAL CLEANED REFIT 1-minute data.

Reads outputs/clean_1min (made by 02_prepare_clean_1min.py). For each house:
  availability / data quality, cycle detection (rule in refit_common.CYCLE_RULE), usage by local hour and day
  of week, cycle characteristics (duration, peak power, energy), threshold sensitivity, peak-window share.
Writes outputs/eda/: availability.csv, cycles.csv, sensitivity.csv, hourly_starts.csv, dow_rate.csv,
eda_summary.json and four figures.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from refit_common import (CYCLE_RULE, HOT_W, OUT_1MIN, OUT_EDA, PEAK_WINDOW, detect_cycles, jdump, jload,
                          load_house_1min, local_naive, nan_runs)

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def main():
    OUT_EDA.mkdir(parents=True, exist_ok=True)
    mapping, meta = jload(OUT_1MIN / "wm_mapping.json"), jload(OUT_1MIN / "meta.json")
    H = {}
    for h, info in mapping.items():
        df = load_house_1min(int(h))
        H[int(h)] = dict(df=df, ch=info["channel"], conv=meta[h]["time_convention"], status=info["status"])
    if len(H) < 4:
        print(f"WARNING: only {len(H)} houses available; the assignment asks for at least 4.")

    avail, cyc_all, summary, sens = [], [], {"houses": {}, "rule": CYCLE_RULE, "peak_window": PEAK_WINDOW}, []
    hourly, dowrate = {}, {}
    for h, d in H.items():
        df, ch = d["df"], d["ch"]
        x = df[ch].to_numpy(dtype="float64")
        agg = df["Aggregate"].to_numpy(dtype="float64")
        loc = local_naive(df.index, d["conv"])
        d["loc"] = loc
        valid = ~np.isnan(x)
        n_valid = int(valid.sum())
        first, last = df.index[valid][0], df.index[valid][-1]
        span_days = (last - first).total_seconds() / 86400
        s_, ln_ = nan_runs(np.isnan(x))
        inside = (s_ > np.flatnonzero(valid)[0]) & (s_ + ln_ <= np.flatnonzero(valid)[-1])
        ln_in = ln_[inside]
        cyc = detect_cycles(x, **CYCLE_RULE)
        comp = cyc[~cyc.has_nan].copy()
        d["cyc"] = comp
        valid_days = n_valid / 1440.0
        date = pd.Series(loc.normalize(), index=df.index)
        cov = pd.Series(valid, index=df.index).groupby(date.values).mean()
        full_days = cov.index[cov >= 0.8]
        d["full_days"] = full_days
        comp["start_local"] = loc[comp.start_idx.to_numpy()] if len(comp) else pd.DatetimeIndex([])
        comp["start_utc"] = df.index[comp.start_idx.to_numpy()] if len(comp) else pd.DatetimeIndex([])
        if len(comp):
            comp["heating_wh"] = [float(np.nansum(np.where(x[s:e + 1] >= HOT_W, x[s:e + 1], 0.0)) / 60.0)
                                  for s, e in zip(comp.start_idx, comp.end_idx)]
            comp["hot"] = comp.peak_w >= HOT_W
        else:
            comp["heating_wh"], comp["hot"] = [], []
        comp["house"] = h
        comp["hour"] = comp.start_local.dt.hour if len(comp) else []
        comp["dow"] = comp.start_local.dt.dayofweek if len(comp) else []
        comp["date"] = comp.start_local.dt.normalize() if len(comp) else []
        cyc_all.append(comp)

        # sensitivity to the ON threshold
        for thr in (10.0, 20.0, 50.0):
            rule = dict(CYCLE_RULE, on_thr=thr)
            c2 = detect_cycles(x, **rule)
            sens.append(dict(house=h, on_thr_w=thr, cycles_complete=int((~c2.has_nan).sum()), cycles_total=len(c2)))

        # hour-of-day and day-of-week
        hrs = np.bincount(comp.hour.astype(int), minlength=24) if len(comp) else np.zeros(24)
        hourly[h] = hrs / max(hrs.sum(), 1)
        fd = pd.DataFrame({"date": full_days})
        fd["dow"] = fd.date.dt.dayofweek
        per_day = comp.groupby("date").size() if len(comp) else pd.Series(dtype=int)
        fd["n"] = fd.date.map(per_day).fillna(0)
        dowrate[h] = fd.groupby("dow").n.mean().reindex(range(7))

        hour_arr = np.asarray(loc.hour)
        in_peak = (hour_arr >= PEAK_WINDOW[0]) & (hour_arr < PEAK_WINDOW[1])
        e_total = float(np.nansum(x))
        e_peak = float(np.nansum(np.where(in_peak, x, np.nan)))
        daily_wh = pd.Series(np.nan_to_num(x), index=df.index).groupby(date.values).sum() / 60.0
        mean_daily_wh = float(daily_wh.reindex(full_days).mean()) if len(full_days) else np.nan
        wk = dowrate[h]
        agg_valid = ~np.isnan(agg)
        row = dict(house=h, channel=ch, mapping_status=d["status"], first_valid=str(first), last_valid=str(last),
                   span_days=round(span_days, 1), minutes_in_span=int(len(df)),
                   valid_minutes=n_valid, missing_pct=round(100 * (1 - n_valid / len(df)), 2),
                   missing_gaps_total=int(len(ln_in)), missing_gaps_ge_60min=int((ln_in >= 60).sum()),
                   longest_gap_hours=round(float(ln_in.max()) / 60, 1) if len(ln_in) else 0.0,
                   zero_pct=round(100 * float((x[valid] == 0).mean()), 2),
                   off_pct=round(100 * float((x[valid] <= CYCLE_RULE["on_thr"]).mean()), 2),
                   aggregate_missing_pct=round(100 * float(1 - agg_valid.mean()), 2),
                   issues_flag_mean=round(float(df["issues_frac"].mean()), 4) if "issues_frac" in df else None,
                   cycles_detected_total=len(cyc), cycles_complete=len(comp), cycles_with_nan=int(cyc.has_nan.sum()),
                   full_days=int(len(full_days)), cycles_per_week=round(len(comp) / (valid_days / 7), 2) if valid_days else None,
                   median_duration_min=float(comp.duration_min.median()) if len(comp) else None,
                   median_peak_w=float(comp.peak_w.median()) if len(comp) else None,
                   median_energy_wh=float(comp.energy_wh.median()) if len(comp) else None,
                   mean_energy_wh=float(comp.energy_wh.mean()) if len(comp) else None,
                   energy_share_peak_window=e_peak / e_total if e_total > 0 else None,
                   cycle_start_share_peak_window=float(hourly[h][PEAK_WINDOW[0]:PEAK_WINDOW[1]].sum()),
                   top_start_hours=[int(i) for i in np.argsort(-hourly[h])[:3]],
                   hot_cycle_share=float(comp.hot.mean()) if len(comp) else None,
                   heating_energy_share=float(comp.heating_wh.sum() / comp.energy_wh.sum()) if len(comp) else None,
                   median_energy_hot_wh=float(comp[comp.hot].energy_wh.median()) if len(comp) and comp.hot.any() else None,
                   median_energy_nonhot_wh=float(comp[~comp.hot].energy_wh.median()) if len(comp) and (~comp.hot).any() else None,
                   mean_daily_wm_wh=mean_daily_wh,
                   est_annual_kwh=mean_daily_wh * 365 / 1000 if mean_daily_wh == mean_daily_wh else None,
                   cycles_per_day_weekday=float(wk.loc[0:4].mean()), cycles_per_day_weekend=float(wk.loc[5:6].mean()))
        avail.append(row)
        summary["houses"][str(h)] = row
        print(f"House {h} ({ch}): {len(comp)} complete cycles, {row['cycles_per_week']} / week, "
              f"{n_valid:,} valid min, missing {row['missing_pct']}%")

    A = pd.DataFrame(avail)
    A.to_csv(OUT_EDA / "availability.csv", index=False)
    C = pd.concat(cyc_all, ignore_index=True)
    C.drop(columns=["start_idx", "end_idx", "has_nan"]).to_csv(OUT_EDA / "cycles.csv", index=False)
    pd.DataFrame(sens).to_csv(OUT_EDA / "sensitivity.csv", index=False)
    pd.DataFrame(hourly).rename_axis("hour").to_csv(OUT_EDA / "hourly_starts.csv")
    pd.DataFrame({h: v for h, v in dowrate.items()}).rename(index=dict(enumerate(DOW))).to_csv(OUT_EDA / "dow_rate.csv")
    summary["sensitivity"] = sens
    summary["hot_threshold_w"] = HOT_W
    summary["total_cycles_complete"] = int(len(C))
    jdump(summary, OUT_EDA / "eda_summary.json")

    # ------------------------------------------------------------------ figures
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # (i) representative 24 h aggregate + WM trace
    best = max(H, key=lambda k: len(H[k]["cyc"]))
    d = H[best]
    c = d["cyc"]
    days_with = c.groupby("date").size().index.intersection(d["full_days"])
    if len(days_with):
        edays = {dd: float(c[c.date == dd].energy_wh.sum()) for dd in days_with}
        med = np.median(list(edays.values()))
        day = min(edays, key=lambda k: abs(edays[k] - med))
        m = (d["loc"].normalize() == day)
        xl = d["loc"][m]
        fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
        ax[0].plot(xl, d["df"]["Aggregate"][m], lw=0.9, color="tab:blue")
        ax[0].set_ylabel("Aggregate (W)")
        ax[1].plot(xl, d["df"][d["ch"]][m], lw=0.9, color="tab:orange")
        ax[1].axhline(CYCLE_RULE["on_thr"], ls="--", lw=0.8, color="gray")
        for r in c[c.date == day].itertuples():
            ax[1].axvspan(d["loc"][r.start_idx], d["loc"][r.end_idx], color="tab:orange", alpha=0.15)
        ax[1].set_ylabel(f"Washing machine, {d['ch']} (W)")
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax[1].set_xlabel("local time")
        fig.suptitle(f"House {best}: 24-hour aggregate and washing-machine trace ({day.date()}); shaded = detected cycle")
        fig.tight_layout()
        fig.savefig(OUT_EDA / "fig_day_trace.png", dpi=110)
        plt.close(fig)
        summary["representative_day"] = dict(house=best, date=str(day.date()))

    # (ii) detected cycles, one per house
    n = len(H)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(11, 3.1 * ((n + 1) // 2)), squeeze=False)
    for ax, (h, d) in zip(axes.ravel(), H.items()):
        c = d["cyc"]
        if not len(c):
            ax.set_title(f"House {h}: no cycles detected")
            continue
        r = c.iloc[(c.energy_wh - c.energy_wh.median()).abs().argsort().iloc[0]]
        s0, e0 = int(r.start_idx), int(r.end_idx)
        lo, hi = max(s0 - 30, 0), min(e0 + 30, len(d["df"]) - 1)
        t = (np.arange(lo, hi + 1) - s0)
        ax.plot(t, d["df"][d["ch"]].to_numpy()[lo:hi + 1], color=colors[h % 10 - 1])
        ax.axvspan(0, e0 - s0, alpha=0.12, color="gray")
        ax.axhline(CYCLE_RULE["on_thr"], ls="--", lw=0.8, color="gray")
        ax.set_title(f"House {h}: {int(r.duration_min)} min, peak {r.peak_w:.0f} W, {r.energy_wh:.0f} Wh", fontsize=10)
        ax.set_xlabel("minutes from cycle start")
        ax.set_ylabel("W")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Detected washing-machine cycles (median-energy cycle per house; grey = detected extent)")
    fig.tight_layout()
    fig.savefig(OUT_EDA / "fig_cycles.png", dpi=110)
    plt.close(fig)

    # (iii) cross-house comparison of cycle characteristics
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, col, lab in zip(ax, ["duration_min", "peak_w", "energy_wh"], ["Duration (min)", "Peak power (W)", "Energy per cycle (Wh)"]):
        data = [H[h]["cyc"][col].to_numpy() for h in H]
        a.boxplot(data, showfliers=False)
        a.set_xticks(range(1, len(H) + 1))
        a.set_xticklabels([f"H{h}\n(n={len(v)})" for h, v in zip(H, data)])
        a.set_ylabel(lab)
        a.set_title(lab)
    fig.suptitle("Cross-house comparison of detected cycle characteristics (outliers hidden)")
    fig.tight_layout()
    fig.savefig(OUT_EDA / "fig_cross_house.png", dpi=110)
    plt.close(fig)

    # (iv) hour-of-day and day-of-week
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for i, h in enumerate(H):
        ax[0].plot(range(24), hourly[h] * 100, marker="o", ms=3, label=f"House {h}")
    ax[0].axvspan(PEAK_WINDOW[0], PEAK_WINDOW[1], color="red", alpha=0.08, label="evening window")
    ax[0].set_xlabel("local hour of cycle start")
    ax[0].set_ylabel("% of cycles")
    ax[0].set_title("Cycle starts by hour of day")
    ax[0].legend(fontsize=8)
    w = 0.8 / len(H)
    for i, h in enumerate(H):
        ax[1].bar(np.arange(7) + i * w, dowrate[h].to_numpy(), width=w, label=f"House {h}")
    ax[1].set_xticks(np.arange(7) + 0.4 - w / 2)
    ax[1].set_xticklabels(DOW)
    ax[1].set_ylabel("cycles per observed day")
    ax[1].set_title("Mean cycles per day by weekday (days with >=80% coverage)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_EDA / "fig_hour_dow.png", dpi=110)
    plt.close(fig)
    print(f"\nEDA outputs written to {OUT_EDA}")


if __name__ == "__main__":
    main()
