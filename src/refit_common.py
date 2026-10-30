"""Shared helpers for the REFIT take-home pipeline (paths, constants, streaming IO, cycle detection)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- paths
# Project root = parent of src/. Override with env vars if your data lives elsewhere.
ROOT = Path(os.environ.get("REFIT_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = Path(os.environ.get("REFIT_RAW_DIR", ROOT / "refit"))
CLEAN_DIR = Path(os.environ.get("REFIT_CLEAN_DIR", ROOT / "clean"))
OUT = ROOT / "outputs"
OUT_RAW = OUT / "raw_house1"
OUT_1MIN = OUT / "clean_1min"
OUT_EDA = OUT / "eda"
OUT_MODEL = OUT / "model"
OUT_COARSE = OUT / "coarse"

# ----------------------------------------------------------------------------- constants
APPLIANCES = [f"Appliance{i}" for i in range(1, 10)]
CHANNELS = ["Aggregate"] + APPLIANCES
TIME_FMT = "%Y-%m-%d %H:%M:%S"
LOCAL_TZ = "Europe/London"
POWER_CAP_W = 25_000.0  # impossible-value cap (justified in DATA.md; see audit threshold table)

# Washing-machine channels as provided by the candidate; verified by 02_prepare_clean_1min.py
WM_MAP_CLAIMED = {1: "Appliance5", 2: "Appliance2", 3: "Appliance6", 5: "Appliance3"}

# Active-cycle rule (plain-language description is in RESULTS.md)
CYCLE_RULE = dict(on_thr=20.0, max_gap=10, min_dur=15, min_peak=100.0, min_energy_wh=20.0)
HOT_W = 1000.0  # a cycle with peak >= this is treated as a "hot" (heating-element) wash; assumption
PEAK_WINDOW = (18, 22)  # illustrative evening window [18:00, 22:00) local; align with the DISCOM tariff


# ----------------------------------------------------------------------------- json
def _sanitize(o):
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Timedelta, Path)):
        return str(o)
    return o


def jdump(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), indent=2), encoding="utf-8")


def jload(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the earlier pipeline stages first.")
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- columns / time
def std_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise header names (case/space insensitive) to Time, Unix, Aggregate, ApplianceN, Issues."""
    ren = {}
    for c in df.columns:
        k = str(c).strip().lower().replace(" ", "")
        if k in ("time", "unix", "aggregate", "issues"):
            ren[c] = k.capitalize() if k != "unix" else "Unix"
        elif k.startswith("appliance") and k[9:].isdigit():
            ren[c] = f"Appliance{int(k[9:])}"
    df = df.rename(columns=ren)
    for need in ("Unix", "Aggregate"):
        if need not in df.columns:
            raise ValueError(f"Column '{need}' not found. Columns: {list(df.columns)}")
    return df


def local_naive(idx_utc: pd.DatetimeIndex, convention: str = "london") -> pd.DatetimeIndex:
    """Local wall-clock (naive) index for hour-of-day analysis."""
    if convention == "utc":
        return idx_utc.tz_localize(None)
    return idx_utc.tz_convert(LOCAL_TZ).tz_localize(None)


def _time_votes(chunk: pd.DataFrame, n=5000):
    if "Time" not in chunk.columns:
        return dict(n=0, london=0, utc=0)
    s = chunk.head(n)
    t = pd.to_datetime(s["Time"], format=TIME_FMT, errors="coerce")
    u = pd.to_numeric(s["Unix"], errors="coerce")
    ok = (t.notna() & u.notna()).to_numpy()
    if not ok.any():
        return dict(n=0, london=0, utc=0)
    uu = pd.to_datetime(u[ok], unit="s", utc=True)
    lon = ((uu.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None) - t[ok]).abs().dt.total_seconds() <= 2).sum()
    utc = ((uu.dt.tz_localize(None) - t[ok]).abs().dt.total_seconds() <= 2).sum()
    return dict(n=int(ok.sum()), london=int(lon), utc=int(utc))


# ----------------------------------------------------------------------------- streaming 1-min resample
def stream_minute_means(path: Path, chunksize: int = 500_000, cap: float = POWER_CAP_W):
    """Chunked read of a REFIT-format CSV -> 1-minute mean frame on a complete UTC minute grid.

    Negative or > cap power values become NaN (counted). NaN minutes remain NaN (no gap filling here).
    Returns (frame, meta). Order of rows in the file is irrelevant (sum/count accumulation).
    """
    parts_s, parts_c = [], []
    meta = dict(rows=0, n_negative=0, n_over_cap=0, columns=None)
    votes = dict(n=0, london=0, utc=0)
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype={"Time": str}, skipinitialspace=True):
        chunk.columns = chunk.columns.astype(str).str.strip()
        chunk = std_columns(chunk)
        v = _time_votes(chunk)
        for k in votes:
            votes[k] += v[k]
        chans = [c for c in CHANNELS if c in chunk.columns]
        has_issues = "Issues" in chunk.columns
        u = pd.to_numeric(chunk["Unix"], errors="coerce")
        ok = u.notna().to_numpy()
        meta["rows"] += int(len(chunk))
        vals = chunk.loc[ok, chans + (["Issues"] if has_issues else [])].apply(pd.to_numeric, errors="coerce")
        vals = vals.astype("float64")
        for c in chans:
            col = vals[c]
            neg, big = col < 0, col > cap
            meta["n_negative"] += int(neg.sum())
            meta["n_over_cap"] += int(big.sum())
            vals[c] = col.mask(neg | big)
        minute = (u[ok] // 60).astype("int64")
        g = vals.groupby(minute)
        s, c_ = g.sum(), g.count()
        size = minute.groupby(minute).size()
        s["n_readings"] = size
        c_["n_readings"] = size
        parts_s.append(s)
        parts_c.append(c_)
        meta["columns"] = chans + (["Issues"] if has_issues else [])
    S = pd.concat(parts_s).groupby(level=0).sum()
    C = pd.concat(parts_c).groupby(level=0).sum()
    full = np.arange(S.index.min(), S.index.max() + 1)
    S, C = S.reindex(full).fillna(0.0), C.reindex(full).fillna(0.0)
    out = S.where(C > 0) / C.where(C > 0)
    out["n_readings"] = S["n_readings"].astype("int32")
    if "Issues" in out.columns:
        out = out.rename(columns={"Issues": "issues_frac"})
    out.index = pd.to_datetime(full * 60, unit="s", utc=True)
    out.index.name = "timestamp_utc"
    meta["time_convention"] = "utc" if votes["utc"] > votes["london"] else "london"
    meta["time_match_london"] = votes["london"] / votes["n"] if votes["n"] else None
    meta["time_match_utc"] = votes["utc"] / votes["n"] if votes["n"] else None
    meta["first_utc"], meta["last_utc"] = str(out.index[0]), str(out.index[-1])
    meta["minutes_in_span"] = int(len(out))
    return out, meta


def load_house_1min(h: int) -> pd.DataFrame:
    p = OUT_1MIN / f"House{h}_1min.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run 02_prepare_clean_1min.py first.")
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp_utc"
    return df


# ----------------------------------------------------------------------------- gap filling
def fill_short_gaps(s: pd.Series, max_len: int):
    """Linearly interpolate NaN runs of <= max_len samples that have valid neighbours on both sides."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    short = isna & (run_len <= max_len)
    interp = s.interpolate(method="linear", limit_area="inside")
    filled = s.where(~short, interp)
    return filled, (short & filled.notna())


def nan_runs(isnan: np.ndarray):
    """Return (starts, lengths) of True-runs in a boolean array."""
    if not isnan.any():
        return np.array([], dtype=int), np.array([], dtype=int)
    edge = np.diff(np.concatenate([[0], isnan.astype(int), [0]]))
    s, e = np.where(edge == 1)[0], np.where(edge == -1)[0]
    return s, e - s


# ----------------------------------------------------------------------------- cycle detection
_CYC_COLS = ["start_idx", "end_idx", "duration_min", "peak_w", "energy_wh", "has_nan"]


def detect_cycles(x, on_thr=20.0, max_gap=10, min_dur=15, min_peak=100.0, min_energy_wh=20.0):
    """Detect active appliance cycles in a 1-minute power array (W).

    A minute is ON if power > on_thr. ON minutes separated by <= max_gap OFF minutes are merged into one
    cycle. A cycle is kept if duration >= min_dur minutes, peak >= min_peak W and energy >= min_energy_wh.
    NaN minutes count as OFF but the cycle is flagged has_nan (excluded from characteristic comparisons).
    """
    x = np.asarray(x, dtype="float64")
    idx = np.flatnonzero(np.nan_to_num(x, nan=0.0) > on_thr)
    if idx.size == 0:
        return pd.DataFrame(columns=_CYC_COLS)
    brk = np.flatnonzero(np.diff(idx) - 1 > max_gap)
    starts = idx[np.r_[0, brk + 1]]
    ends = idx[np.r_[brk, idx.size - 1]]
    rows = []
    for s, e in zip(starts, ends):
        seg = x[s:e + 1]
        z = np.nan_to_num(seg, nan=0.0)
        rows.append((s, e, e - s + 1, z.max(), z.sum() / 60.0, bool(np.isnan(seg).any())))
    df = pd.DataFrame(rows, columns=_CYC_COLS)
    keep = (df.duration_min >= min_dur) & (df.peak_w >= min_peak) & (df.energy_wh >= min_energy_wh)
    return df[keep].reset_index(drop=True)


def signature(x, rule=CYCLE_RULE):
    """Washing-machine-like signature score (0-4) for a 1-min power series. Heuristic, not proof."""
    x = np.asarray(x, dtype="float64")
    valid = ~np.isnan(x)
    n = int(valid.sum())
    if n < 1440:
        return dict(score=0, on_frac=np.nan, cycles_per_week=np.nan, median_duration_min=np.nan, p999_w=np.nan)
    on_frac = float((x[valid] > rule["on_thr"]).mean())
    cyc = detect_cycles(x, **rule)
    cyc = cyc[~cyc.has_nan]
    per_week = len(cyc) / (n / 1440.0 / 7.0)
    med = float(cyc.duration_min.median()) if len(cyc) else np.nan
    p999 = float(np.nanpercentile(x, 99.9))
    checks = [0.005 <= on_frac <= 0.15, 0.5 <= per_week <= 21, (20 <= med <= 180) if med == med else False,
              p999 >= 500]
    return dict(score=int(sum(checks)), on_frac=on_frac, cycles_per_week=per_week,
                median_duration_min=med, p999_w=p999)
