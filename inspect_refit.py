
import os
import glob
import json
from collections import Counter

import pandas as pd

FOLDER = r"C:\medhavi\proj\clean"

print("\n=== REFIT DATASET INSPECTOR ===\n")

files = sorted(glob.glob(os.path.join(FOLDER, "CLEAN_House*.csv")))

if not files:
    print("ERROR: No CLEAN_House*.csv files found.")
    input("Press Enter to exit...")
    raise SystemExit

report = {
    "folder": FOLDER,
    "number_of_files": len(files),
    "households": []
}

for filepath in files:
    name = os.path.basename(filepath)
    print(f"\nInspecting {name}...")

    result = {
        "file": name,
        "size_MB": round(os.path.getsize(filepath) / (1024**2), 2),
        "columns": [],
        "rows": 0,
        "missing": {},
        "negative_values": {},
        "zero_percentage": {},
        "minimum": {},
        "maximum": {},
        "time_start": None,
        "time_end": None,
        "unix_interval_seconds": {},
        "issues_values": {}
    }

    interval_counts = Counter()
    issue_counts = Counter()
    previous_unix = None
    zero_counts = Counter()
    numeric_counts = Counter()

    try:
        for chunk in pd.read_csv(filepath, chunksize=200000):
            if not result["columns"]:
                result["columns"] = chunk.columns.tolist()

                for col in chunk.columns:
                    if col not in ["Time", "Issues"]:
                        result["missing"][col] = 0
                        result["negative_values"][col] = 0
                        result["minimum"][col] = None
                        result["maximum"][col] = None

            result["rows"] += len(chunk)

            if "Time" in chunk.columns:
                times = pd.to_datetime(chunk["Time"], errors="coerce")
                valid_times = times.dropna()

                if len(valid_times):
                    start = str(valid_times.min())
                    end = str(valid_times.max())

                    if result["time_start"] is None or start < result["time_start"]:
                        result["time_start"] = start

                    if result["time_end"] is None or end > result["time_end"]:
                        result["time_end"] = end

            if "Unix" in chunk.columns:
                unix = pd.to_numeric(chunk["Unix"], errors="coerce")
                intervals = unix.diff()

                if previous_unix is not None and len(unix):
                    intervals.iloc[0] = unix.iloc[0] - previous_unix

                interval_counts.update(
                    round(float(x), 3) for x in intervals.dropna()
                )

                valid = unix.dropna()
                if len(valid):
                    previous_unix = valid.iloc[-1]

            for col in result["missing"]:
                values = pd.to_numeric(chunk[col], errors="coerce")

                result["missing"][col] += int(values.isna().sum())
                result["negative_values"][col] += int((values < 0).sum())

                valid = values.dropna()

                if len(valid):
                    zero_counts[col] += int((valid == 0).sum())
                    numeric_counts[col] += len(valid)

                    mn = float(valid.min())
                    mx = float(valid.max())

                    old_min = result["minimum"][col]
                    old_max = result["maximum"][col]

                    result["minimum"][col] = mn if old_min is None else min(old_min, mn)
                    result["maximum"][col] = mx if old_max is None else max(old_max, mx)

            if "Issues" in chunk.columns:
                issue_counts.update(
                    chunk["Issues"].fillna("MISSING").astype(str).value_counts().to_dict()
                )

        for col in result["zero_percentage"]:
            count = numeric_counts[col]
            result["zero_percentage"][col] = (
                round(100 * zero_counts[col] / count, 2)
                if count else None
            )

        result["unix_interval_seconds"] = dict(interval_counts.most_common(10))
        result["issues_values"] = dict(issue_counts.most_common(15))

        report["households"].append(result)

        print("Rows:", result["rows"])
        print("Size:", result["size_MB"], "MB")
        print("Time:", result["time_start"], "to", result["time_end"])
        print("Intervals:", result["unix_interval_seconds"])

    except Exception as e:
        print("ERROR:", e)
        result["error"] = str(e)
        report["households"].append(result)

output = os.path.join(FOLDER, "REFIT_DIAGNOSTIC_REPORT.json")

with open(output, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n===================================")
print("INSPECTION COMPLETE")
print("Households found:", len(files))
print("Report saved:", output)
print("===================================")

input("Press Enter to exit...")
