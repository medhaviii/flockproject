#!/usr/bin/env python
"""
05_coarse_resolution.py -- what happens to washing-machine disaggregation with 15- or 30-minute meter data?

Simulates a coarse smart meter by averaging the aggregate to R-minute blocks (R = 15, 30) and predicts the
washing machine's mean power per block. Uses the SAME chronological split boundaries as 04_model_nilm.py
(plus an embargo of K+1 blocks) so train/test do not overlap.

Compared on the identical set of test blocks:
  always-off, hour-of-week profile, HGB trained on R-minute data, and the 1-minute model from stage 4 whose
  predictions are averaged into R-minute blocks (this shows what is lost by coarsening the INPUT).
Also reports how many detected cycles are shorter than 15 / 30 / 60 minutes (cycle shape cannot be resolved).
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingRegressor

from refit_common import CYCLE_RULE, OUT_1MIN, OUT_COARSE, OUT_EDA, OUT_MODEL, jdump, jload, load_house_1min, local_naive

ON_THR = CYCLE_RULE["on_thr"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    on_t, on_p = y > ON_THR, p > ON_THR
    tp = int((on_t & on_p).sum())
    prec = tp / on_p.sum() if on_p.sum() else 0.0
    rec = tp / on_t.sum() if on_t.sum() else 0.0
    return dict(n_blocks=int(len(y)), mae_w=float(np.abs(err).mean()), rmse_w=float(np.sqrt((err ** 2).mean())),
                sae_pct=float(100 * (p.sum() - y.sum()) / y.sum()) if y.sum() > 0 else None,
                on_f1=float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0)


def block_table(df, ch, conv, R, K):
    m_abs = np.asarray((df.index - EPOCH) // pd.Timedelta(minutes=1))
    bid = m_abs // R
    g = pd.DataFrame({"a": df["Aggregate"].to_numpy(), "y": df[ch].to_numpy(), "b": bid}).groupby("b")
    mean, cnt = g.mean(), g.count()
    ok = (cnt["a"] == R) & (cnt["y"] == R)
    ids = np.arange(mean.index.min(), mean.index.max() + 1)
    a = mean["a"].where(ok).reindex(ids).to_numpy(dtype="float64")
    y = mean["y"].where(ok).reindex(ids).to_numpy(dtype="float64")
    padded = np.concatenate([np.full(K, np.nan), a, np.full(K, np.nan)])
    win = sliding_window_view(padded, 2 * K + 1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = np.stack([np.nanmean(win, 1), np.nanstd(win, 1), np.nanmin(win, 1), np.nanmax(win, 1),
                          a - np.nanmin(win, 1)], axis=1)
    ts = EPOCH + pd.to_timedelta(ids * R + R / 2.0, unit="m")
    loc = local_naive(pd.DatetimeIndex(ts), conv)
    hr = np.asarray(loc.hour) + np.asarray(loc.minute) / 60.0
    tf = np.stack([np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), (np.asarray(loc.dayofweek) >= 5)], axis=1)
    X = np.hstack([win, stats, tf]).astype("float32")
    how = np.asarray(loc.dayofweek) * 24 + np.asarray(loc.hour)
    return dict(ids=ids, a=a, y=y, X=X, how=how, start_min=ids * R)


def main():
    OUT_COARSE.mkdir(parents=True, exist_ok=True)
    mapping, meta = jload(OUT_1MIN / "wm_mapping.json"), jload(OUT_1MIN / "meta.json")
    bounds = jload(OUT_MODEL / "split_bounds.json")
    TP = pd.read_csv(OUT_MODEL / "test_predictions.csv.gz")
    TP["timestamp_utc"] = pd.to_datetime(TP["timestamp_utc"], utc=True)
    TP["m_abs"] = np.asarray((TP["timestamp_utc"] - EPOCH) // pd.Timedelta(minutes=1))
    houses = [int(h) for h in mapping if h in bounds]
    results, rows = {}, []
    for R, K in ((15, 4), (30, 2)):
        Eb = (K + 1) * R
        T = {}
        for h in houses:
            df = load_house_1min(h)
            T[h] = block_table(df, mapping[str(h)]["channel"], meta[str(h)]["time_convention"], R, K)
            b = bounds[str(h)]
            sm = T[h]["start_min"]
            valid = ~np.isnan(T[h]["y"]) & ~np.isnan(T[h]["a"])
            T[h]["train"] = valid & (sm < b["boundary_minute_train_val"] - Eb)
            T[h]["test"] = valid & (sm >= b["boundary_minute_val_test"] + Eb)
        Xtr = np.vstack([T[h]["X"][T[h]["train"]] for h in houses])
        ytr = np.concatenate([T[h]["y"][T[h]["train"]] for h in houses])
        model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
                                              l2_regularization=1.0, early_stopping=False, random_state=0).fit(Xtr, ytr)
        ev = {k: [] for k in ["y", "off", "profile", "hgb_coarse", "hgb_1min_aggregated"]}
        for h in houses:
            t = T[h]
            prof = np.array([t["y"][t["train"] & (t["how"] == k)].mean() if (t["train"] & (t["how"] == k)).any() else 0.0
                             for k in range(168)])
            tp = TP[TP.house == h].copy()
            tp["bid"] = tp["m_abs"] // R
            g = tp.groupby("bid").agg(n=("m_abs", "size"), a=("actual_w", "mean"), p=("hgb_pooled", "mean"))
            g = g[g.n == R]
            idx = np.flatnonzero(t["test"])
            pos = pd.Series(idx, index=t["ids"][idx])
            common = pos.index.intersection(g.index)
            if len(common) == 0:
                continue
            ii = pos.loc[common].to_numpy()
            ev["y"].append(t["y"][ii])
            ev["off"].append(np.zeros(len(ii)))
            ev["profile"].append(prof[t["how"][ii]])
            ev["hgb_coarse"].append(np.clip(model.predict(t["X"][ii]), 0, None))
            ev["hgb_1min_aggregated"].append(g.loc[common, "p"].to_numpy())
        if not ev["y"]:
            print(f"R={R}: no overlapping test blocks - skipped")
            continue
        y = np.concatenate(ev["y"])
        results[str(R)] = {}
        for name in ["off", "profile", "hgb_coarse", "hgb_1min_aggregated"]:
            r = metrics(y, np.concatenate(ev[name]))
            results[str(R)][name] = r
            rows.append(dict(resolution_min=R, model=name, **r))
        print(f"R={R} min: {len(y):,} common test blocks")
    pd.DataFrame(rows).to_csv(OUT_COARSE / "coarse_metrics.csv", index=False)

    cyc = pd.read_csv(OUT_EDA / "cycles.csv")
    dur = cyc.duration_min
    shape = dict(n_cycles=int(len(cyc)), median_duration_min=float(dur.median()),
                 **{f"share_shorter_than_{t}min": float((dur < t).mean()) for t in (15, 30, 60)})
    jdump(dict(results=results, cycle_duration_vs_resolution=shape, blocks_used=dict(
        K_blocks={"15": 4, "30": 2}, note="aggregate features = +-1 h of R-minute block means")), OUT_COARSE / "coarse_results.json")

    # ---- figure: example cycle at 1/15/30 min + MAE bars
    h0 = houses[0]
    df = load_house_1min(h0)
    ch = mapping[str(h0)]["channel"]
    c0 = cyc[cyc.house == h0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    if len(c0):
        r = c0.iloc[(c0.energy_wh - c0.energy_wh.median()).abs().argsort().iloc[0]]
        t0 = pd.Timestamp(r.start_utc)
        w = df.loc[t0 - pd.Timedelta(hours=1): t0 + pd.Timedelta(minutes=int(r.duration_min) + 60)]
        ax[0].plot(w.index, w["Aggregate"], lw=0.9, color="tab:blue", label="Aggregate, 1-min")
        ax[0].plot(w.index, w[ch], lw=0.9, color="tab:orange", label="Washing machine, 1-min")
        for R, col in ((15, "k"), (30, "tab:red")):
            m = (np.asarray((w.index - EPOCH) // pd.Timedelta(minutes=1)) // R)
            ag = w["Aggregate"].groupby(m).transform("mean")
            ax[0].step(w.index, ag, where="post", color=col, lw=1.4, ls="--", label=f"Aggregate, {R}-min mean")
        ax[0].legend(fontsize=7)
        ax[0].set_title(f"House {h0}: one {int(r.duration_min)}-min cycle seen at 1, 15 and 30 min")
        ax[0].set_ylabel("W")
        ax[0].tick_params(axis="x", labelrotation=30)
    M = pd.DataFrame(rows)
    if len(M):
        names = {"off": "Always-off", "profile": "Profile", "hgb_coarse": "HGB (R-min data)", "hgb_1min_aggregated": "HGB (1-min data, averaged)"}
        for i, (k, lab) in enumerate(names.items()):
            vals = [float(M[(M.resolution_min == R) & (M.model == k)].mae_w.iloc[0]) if len(M[(M.resolution_min == R) & (M.model == k)]) else np.nan
                    for R in (15, 30)]
            ax[1].bar(np.arange(2) + i * 0.2, vals, 0.2, label=lab)
        ax[1].set_xticks(np.arange(2) + 0.3)
        ax[1].set_xticklabels(["15-min blocks", "30-min blocks"])
        ax[1].set_ylabel("MAE of block-mean washing-machine power (W)")
        ax[1].set_title("Test error at coarse resolution (same blocks for all models)")
        ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_COARSE / "fig_coarse.png", dpi=110)
    plt.close(fig)
    print(M.round(3).to_string(index=False))
    print(f"\nOutputs in {OUT_COARSE}")


if __name__ == "__main__":
    main()
