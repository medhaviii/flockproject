#!/usr/bin/env python
"""
04_model_nilm.py -- washing-machine disaggregation from 1-minute aggregate power (official cleaned REFIT).

Task      : predict washing-machine power (W) at minute t from the aggregate signal in [t-15, t+15] min
            (a Seq2Point-style window; centred window = non-causal / offline disaggregation) + hour/weekend.
Models    : (1) always-off baseline, (2) hour-of-week usage-profile baseline (train mean per house),
            (3) HistGradientBoostingRegressor trained on all houses' training periods ("pooled"),
            (4) the same model trained WITHOUT the target house (leave-one-house-out, tests transfer to an
                unseen home).
Split     : per house, CHRONOLOGICAL by valid rows: first 70 % train, next 10 % validation, last 20 % test.
            An embargo of 16 min on both sides of each boundary removes rows whose 31-min input window could
            touch the neighbouring segment, so no window overlaps across splits. The validation block is used
            only to choose the near-zero clipping threshold tau. The test block is touched once, at the end.
Metrics   : MAE, RMSE (W), signal-aggregate error of energy (SAE, %), ON-state F1 at 20 W, and cycle-level
            recall/precision using the same cycle rule as the EDA.
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingRegressor

from refit_common import CYCLE_RULE, OUT_1MIN, OUT_MODEL, detect_cycles, jdump, jload, load_house_1min, local_naive

W = 15
E = W + 1
ON_THR = CYCLE_RULE["on_thr"]
TAUS = [0, 5, 10, 20, 30, 50]
MODELS = ["always_off", "hour_of_week_profile", "hgb_pooled", "hgb_leave_one_house_out"]
NICE = {"always_off": "Always-off", "hour_of_week_profile": "Hour-of-week profile",
        "hgb_pooled": "HGB (pooled)", "hgb_leave_one_house_out": "HGB (leave-one-house-out)"}


def build_features(agg: np.ndarray, loc: pd.DatetimeIndex):
    a = agg.astype(np.float32)
    n = len(a)
    padded = np.concatenate([np.full(W, np.nan, np.float32), a, np.full(W, np.nan, np.float32)])
    win = sliding_window_view(padded, 2 * W + 1)
    with np.errstate(all="ignore"):
        feats = [win, win.mean(1, keepdims=True), win.std(1, keepdims=True), win.min(1, keepdims=True),
                 win.max(1, keepdims=True), np.median(win, axis=1, keepdims=True),
                 (win[:, W] - win.min(1))[:, None], (win[:, W] - np.median(win, axis=1))[:, None],
                 (win[:, W] - win[:, W - 1])[:, None], (win[:, W + 1] - win[:, W])[:, None]]
    hr = (np.asarray(loc.hour) + np.asarray(loc.minute) / 60.0).astype(np.float32)
    tf = np.stack([np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
                   (np.asarray(loc.dayofweek) >= 5).astype(np.float32)], axis=1)
    X = np.hstack(feats + [tf]).astype(np.float32)
    names = [f"agg_{k:+d}" for k in range(-W, W + 1)] + ["win_mean", "win_std", "win_min", "win_max", "win_median",
                                                          "agg_minus_min", "agg_minus_median", "d_back", "d_fwd",
                                                          "hour_sin", "hour_cos", "weekend"]
    return X, names


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    on_t, on_p = y > ON_THR, p > ON_THR
    tp = int((on_t & on_p).sum())
    prec = tp / on_p.sum() if on_p.sum() else 0.0
    rec = tp / on_t.sum() if on_t.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(n=int(len(y)), mae_w=float(np.abs(err).mean()), rmse_w=float(np.sqrt((err ** 2).mean())),
                sae_pct=float(100 * (p.sum() - y.sum()) / y.sum()) if y.sum() > 0 else None,
                on_f1=f1, on_precision=prec, on_recall=rec, mean_actual_w=float(y.mean()),
                mean_pred_w=float(p.mean()))


def cycle_level(y_full, p_full):
    ct = detect_cycles(y_full, **CYCLE_RULE)
    ct = ct[~ct.has_nan].reset_index(drop=True)
    cp = detect_cycles(p_full, **CYCLE_RULE)
    cp = cp[~cp.has_nan].reset_index(drop=True)
    ps, pe = cp.start_idx.to_numpy(), cp.end_idx.to_numpy()
    ts, te = ct.start_idx.to_numpy(), ct.end_idx.to_numpy()
    mt = np.array([np.any((ps <= e) & (pe >= s)) for s, e in zip(ts, te)], dtype=bool) if len(ct) else np.array([], bool)
    mp = np.array([np.any((ts <= e) & (te >= s)) for s, e in zip(ps, pe)], dtype=bool) if len(cp) else np.array([], bool)
    ratio = [np.nansum(p_full[s:e + 1]) / np.nansum(y_full[s:e + 1]) for s, e in zip(ts, te)] if len(ct) else []
    return dict(true_cycles=int(len(ct)), detected_true_cycles=int(mt.sum()),
                recall=float(mt.mean()) if len(mt) else None, predicted_cycles=int(len(cp)),
                true_positive_pred_cycles=int(mp.sum()), false_cycles=int((~mp).sum()),
                precision=float(mp.mean()) if len(mp) else None,
                median_energy_ratio_pred_over_true=float(np.median(ratio)) if len(ratio) else None)


def fit(X, y, args):
    m = HistGradientBoostingRegressor(max_iter=args.max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                      min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
                                      random_state=args.seed)
    return m.fit(X, y)


def predict(m, X, tau):
    p = np.clip(m.predict(X), 0, None)
    p[p < tau] = 0.0
    return p


def choose_tau(m, Xv, yv):
    res = {t: float(np.abs(predict(m, Xv, t) - yv).mean()) for t in TAUS}
    return min(res, key=lambda k: (res[k], k)), res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-train-per-house", type=int, default=200_000)
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    mapping, meta = jload(OUT_1MIN / "wm_mapping.json"), jload(OUT_1MIN / "meta.json")

    D, bounds, feat_names = {}, {}, None
    for h, info in mapping.items():
        t0 = time.time()
        h = int(h)
        df = load_house_1min(h)
        loc = local_naive(df.index, meta[str(h)]["time_convention"])
        y = df[info["channel"]].to_numpy(dtype="float32")
        agg = df["Aggregate"].to_numpy(dtype="float32")
        X, feat_names = build_features(agg, loc)
        valid = ~np.isnan(X[:, :2 * W + 1]).any(axis=1) & ~np.isnan(y)
        pos = np.flatnonzero(valid)
        if len(pos) < 5000:
            print(f"House {h}: too few valid rows ({len(pos)}) - skipped")
            continue
        tb1, tb2 = int(pos[int(0.7 * len(pos))]), int(pos[int(0.8 * len(pos))])
        t = np.arange(len(df))
        seg = {"train": valid & (t < tb1 - E), "val": valid & (t >= tb1 + E) & (t < tb2 - E), "test": valid & (t >= tb2 + E)}
        how = (np.asarray(loc.dayofweek) * 24 + np.asarray(loc.hour)).astype(int)
        prof = np.array([y[seg["train"] & (how == k)].mean() if (seg["train"] & (how == k)).any() else 0.0 for k in range(168)])
        tr_idx = np.flatnonzero(seg["train"])
        if len(tr_idx) > args.max_train_per_house:
            tr_idx = np.sort(rng.choice(tr_idx, args.max_train_per_house, replace=False))
        te_idx, va_idx = np.flatnonzero(seg["test"]), np.flatnonzero(seg["val"])
        D[h] = dict(ch=info["channel"], status=info["status"], Xtr=X[tr_idx], ytr=y[tr_idx],
                    Xva=X[va_idx], yva=y[va_idx], Xte=X[te_idx], yte=y[te_idx], te_idx=te_idx,
                    how_te=how[te_idx], prof=prof, index=df.index, loc=loc, n_train_all=int(seg["train"].sum()),
                    n_val=int(len(va_idx)), n_test=int(len(te_idx)))
        bounds[str(h)] = dict(channel=info["channel"], embargo_min=E, window_half_min=W,
                              train_end_utc=str(df.index[tb1 - E]), val_start_utc=str(df.index[tb1 + E]),
                              val_end_utc=str(df.index[tb2 - E]), test_start_utc=str(df.index[tb2 + E]),
                              test_end_utc=str(df.index[te_idx[-1]]), first_valid_utc=str(df.index[pos[0]]),
                              boundary_minute_train_val=int(df.index[tb1].timestamp() // 60),
                              boundary_minute_val_test=int(df.index[tb2].timestamp() // 60),
                              n_valid_rows=int(len(pos)), n_train_rows=int(seg["train"].sum()),
                              n_train_used=int(len(tr_idx)), n_val_rows=int(len(va_idx)), n_test_rows=int(len(te_idx)))
        print(f"House {h} ({info['channel']}): valid {len(pos):,}  train {seg['train'].sum():,} (using {len(tr_idx):,})  "
              f"val {len(va_idx):,}  test {len(te_idx):,}   [{time.time() - t0:.0f}s]")
        del X
    houses = list(D)
    if len(houses) < 2:
        raise SystemExit("Need at least 2 houses with enough data.")

    print("\nTraining pooled model ...")
    t0 = time.time()
    pooled = fit(np.vstack([D[h]["Xtr"] for h in houses]), np.concatenate([D[h]["ytr"] for h in houses]), args)
    tau_p, tau_curve = choose_tau(pooled, np.vstack([D[h]["Xva"] for h in houses]), np.concatenate([D[h]["yva"] for h in houses]))
    print(f"  done in {time.time() - t0:.0f}s; tau chosen on validation = {tau_p} W")

    loho, tau_l = {}, {}
    for h in houses:
        others = [o for o in houses if o != h]
        print(f"Training leave-one-house-out model (excluding House {h}) ...")
        m = fit(np.vstack([D[o]["Xtr"] for o in others]), np.concatenate([D[o]["ytr"] for o in others]), args)
        tau_l[h], _ = choose_tau(m, np.vstack([D[o]["Xva"] for o in others]), np.concatenate([D[o]["yva"] for o in others]))
        loho[h] = m

    # ----------------------------------------------------------------------- evaluate on TEST (once)
    preds, rows = {}, []
    for h in houses:
        d = D[h]
        P = {"always_off": np.zeros(len(d["yte"])), "hour_of_week_profile": d["prof"][d["how_te"]],
             "hgb_pooled": predict(pooled, d["Xte"], tau_p), "hgb_leave_one_house_out": predict(loho[h], d["Xte"], tau_l[h])}
        preds[h] = P
        for mname in MODELS:
            rows.append(dict(house=h, model=mname, **metrics(d["yte"], P[mname])))
    for mname in MODELS:
        yy = np.concatenate([D[h]["yte"] for h in houses])
        pp = np.concatenate([preds[h][mname] for h in houses])
        rows.append(dict(house="ALL", model=mname, **metrics(yy, pp)))
    M = pd.DataFrame(rows)
    M.to_csv(OUT_MODEL / "metrics_by_house.csv", index=False)

    # test predictions file (compact) for plots / report
    recs = []
    for h in houses:
        d = D[h]
        r = pd.DataFrame({"house": h, "timestamp_utc": d["index"][d["te_idx"]], "aggregate_w": d["Xte"][:, W],
                          "actual_w": d["yte"], **{k: v for k, v in preds[h].items()}})
        recs.append(r)
    TP = pd.concat(recs, ignore_index=True)
    numc = TP.select_dtypes("number").columns
    TP[numc] = TP[numc].round(1)
    TP.to_csv(OUT_MODEL / "test_predictions.csv.gz", index=False, compression="gzip")

    # ----------------------------------------------------------------------- cycle-level + failure analysis
    cyc = {}
    for mname in ("hgb_pooled", "hgb_leave_one_house_out", "hour_of_week_profile"):
        cyc[mname] = {}
        for h in houses:
            d = D[h]
            lo, hi = d["te_idx"][0], d["te_idx"][-1]
            yf, pf = np.full(hi - lo + 1, np.nan), np.full(hi - lo + 1, np.nan)
            yf[d["te_idx"] - lo], pf[d["te_idx"] - lo] = d["yte"], preds[h][mname]
            cyc[mname][str(h)] = cycle_level(yf, pf)

    ys = np.concatenate([D[h]["yte"] for h in houses])
    ps = np.concatenate([preds[h]["hgb_pooled"] for h in houses])
    others = np.concatenate([D[h]["Xte"][:, W] - D[h]["yte"] for h in houses])
    on = ys > ON_THR
    by_load = []
    if on.sum() > 30:
        qs = np.quantile(others[on], [0, 1 / 3, 2 / 3, 1])
        for i, lab in enumerate(["low other load", "medium other load", "high other load"]):
            m = on & (others >= qs[i]) & (others <= qs[i + 1])
            by_load.append(dict(bin=lab, other_load_range_w=[float(qs[i]), float(qs[i + 1])], n_on_minutes=int(m.sum()),
                                mae_w=float(np.abs(ps[m] - ys[m]).mean()), mean_actual_w=float(ys[m].mean()),
                                mean_pred_w=float(ps[m].mean())))
    day_rows = []
    for h in houses:
        d = D[h]
        s = pd.DataFrame({"date": d["loc"][d["te_idx"]].normalize(), "y": d["yte"], "p": preds[h]["hgb_pooled"],
                          "pf": preds[h]["hour_of_week_profile"]})
        for date, g in s.groupby("date"):
            if len(g) < 0.9 * 1440:
                continue
            day_rows.append(dict(house=h, date=date, n=len(g), mae_w=float(np.abs(g.p - g.y).mean()),
                                 actual_wh=float(g.y.sum() / 60), pred_wh=float(g.p.sum() / 60),
                                 false_positive_wh=float(np.clip(g.p - g.y, 0, None).sum() / 60),
                                 missed_wh=float(np.clip(g.y - g.p, 0, None).sum() / 60),
                                 profile_mae_w=float(np.abs(g.pf - g.y).mean()), has_cycle=bool(g.y.max() > CYCLE_RULE["min_peak"])))
    DAYS = pd.DataFrame(day_rows)
    DAYS.to_csv(OUT_MODEL / "test_day_errors.csv", index=False)
    worst = DAYS.sort_values("mae_w", ascending=False).head(6)
    worst.to_csv(OUT_MODEL / "failure_cases.csv", index=False)

    def pct_improve(a, b):
        return float(100 * (b - a) / b) if b else None
    imp = {}
    for h in [str(x) for x in houses] + ["ALL"]:
        sub = M[M.house.astype(str) == h].set_index("model")
        base = min(sub.loc["always_off", "mae_w"], sub.loc["hour_of_week_profile", "mae_w"])
        imp[h] = dict(best_baseline_mae_w=float(base),
                      hgb_pooled_mae_improvement_pct=pct_improve(sub.loc["hgb_pooled", "mae_w"], base),
                      hgb_loho_mae_improvement_pct=pct_improve(sub.loc["hgb_leave_one_house_out", "mae_w"], base))
    jdump(dict(
        config=dict(window_half_min=W, embargo_min=E, split="chronological 70/10/20 of valid rows per house",
                    max_train_per_house=args.max_train_per_house, max_iter=args.max_iter, seed=args.seed,
                    on_threshold_w=ON_THR, tau_pooled_w=tau_p, tau_loho_w=tau_l, tau_val_mae_curve=tau_curve,
                    n_features=len(feat_names), features=feat_names, cycle_rule=CYCLE_RULE, models=MODELS),
        splits=bounds, metrics=M.to_dict(orient="records"), improvement_vs_best_baseline=imp, cycle_level=cyc,
        error_by_other_load=by_load, worst_days=worst.to_dict(orient="records"),
        statuses={str(h): D[h]["status"] for h in houses}), OUT_MODEL / "model_results.json")
    jdump(bounds, OUT_MODEL / "split_bounds.json")

    # ----------------------------------------------------------------------- figures
    colors = {"always_off": "0.6", "hour_of_week_profile": "tab:green", "hgb_pooled": "tab:orange",
              "hgb_leave_one_house_out": "tab:purple"}
    n = len(houses)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(13, 3.4 * ((n + 1) // 2)), squeeze=False)
    for ax, h in zip(axes.ravel(), houses):
        dd = DAYS[(DAYS.house == h) & DAYS.has_cycle]
        if dd.empty:
            ax.set_title(f"House {h}: no test day with a cycle")
            continue
        r = dd.iloc[(dd.mae_w - dd.mae_w.median()).abs().argsort().iloc[0]]
        d = D[h]
        m = (d["loc"][d["te_idx"]].normalize() == r.date)
        xl = d["loc"][d["te_idx"]][m]
        ax2 = ax.twinx()
        ax2.plot(xl, d["Xte"][m, W], color="tab:blue", lw=0.6, alpha=0.35)
        ax2.set_ylabel("Aggregate (W)", color="tab:blue", fontsize=8)
        ax.plot(xl, d["yte"][m], color="k", lw=1.2, label="Actual")
        ax.plot(xl, preds[h]["hgb_pooled"][m], color="tab:orange", lw=1.0, label="HGB pooled")
        ax.plot(xl, preds[h]["hour_of_week_profile"][m], color="tab:green", lw=0.9, ls="--", label="Profile baseline")
        ax.set_title(f"House {h} - test day {r.date.date()} (MAE {r.mae_w:.1f} W)", fontsize=10)
        ax.set_ylabel("Washing machine (W)")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Actual vs predicted washing-machine power on held-out TEST days (median-error day with a cycle)")
    fig.tight_layout()
    fig.savefig(OUT_MODEL / "fig_pred_examples.png", dpi=110)
    plt.close(fig)

    wf = worst.head(4)
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.6))
    for ax, r in zip(axes.ravel(), wf.itertuples()):
        d = D[r.house]
        m = (d["loc"][d["te_idx"]].normalize() == r.date)
        xl = d["loc"][d["te_idx"]][m]
        ax2 = ax.twinx()
        ax2.plot(xl, d["Xte"][m, W], color="tab:blue", lw=0.6, alpha=0.35)
        ax.plot(xl, d["yte"][m], color="k", lw=1.2, label="Actual")
        ax.plot(xl, preds[r.house]["hgb_pooled"][m], color="tab:orange", lw=1.0, label="HGB pooled")
        ax.set_title(f"House {r.house}, {r.date.date()}: MAE {r.mae_w:.1f} W; false+ {r.false_positive_wh:.0f} Wh, "
                     f"missed {r.missed_wh:.0f} Wh", fontsize=9)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[len(wf):]:
        ax.axis("off")
    fig.suptitle("Failure cases: the four highest-MAE test days (blue = aggregate, right axis)")
    fig.tight_layout()
    fig.savefig(OUT_MODEL / "fig_failures.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    labels = [str(h) for h in houses] + ["ALL"]
    w = 0.2
    for j, (col, ttl) in enumerate([("mae_w", "MAE (W) - lower is better"), ("rmse_w", "RMSE (W) - lower is better")]):
        for i, mname in enumerate(MODELS):
            vals = [float(M[(M.house.astype(str) == lab) & (M.model == mname)][col].iloc[0]) for lab in labels]
            ax[j].bar(np.arange(len(labels)) + i * w, vals, w, color=colors[mname], label=NICE[mname])
        ax[j].set_xticks(np.arange(len(labels)) + 1.5 * w)
        ax[j].set_xticklabels([f"House {l}" if l != "ALL" else "All" for l in labels])
        ax[j].set_title(ttl)
    ax[0].legend(fontsize=7)
    fig.suptitle("Test-set error by model and house")
    fig.tight_layout()
    fig.savefig(OUT_MODEL / "fig_metrics.png", dpi=110)
    plt.close(fig)

    print("\nTEST metrics (pooled over all test minutes):")
    print(M[M.house == "ALL"][["model", "mae_w", "rmse_w", "sae_pct", "on_f1"]].round(3).to_string(index=False))
    print(f"\nOutputs in {OUT_MODEL}")


if __name__ == "__main__":
    main()
