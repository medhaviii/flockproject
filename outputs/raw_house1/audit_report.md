# Raw House 1 audit and cleaning report (auto-generated)

## 1. Files and coverage

| Part | Rows | First Unix (file order) | Last Unix (file order) |
|---|---|---|---|
| 1 | 4,377,877 | 2013-10-09 14:06:17 UTC | 2014-10-01 01:27:07 UTC |
| 2 | 4,140,886 | 2014-06-27 21:57:49 UTC | 2015-07-10 11:56:32 UTC |

Total raw rows: **8,518,763**. Parts overlap in time: **True**.

## 2. Timestamps, ordering, duplicates

| Check | Count |
|---|---|
| Unix missing | 0 |
| Unix non-numeric | 0 |
| 'Time' string unparseable | 0 |
| Rows cross-checked Time vs Unix | 8,518,763 |
| Time != Unix as Europe/London (>2 s) | 5,048,558 |
| Time != Unix as UTC (>2 s) | 0 |
| Backward time steps (out of order) | 2 |
| Adjacent rows with identical Unix | 1,018,705 |
|   of which exact duplicates | 883,751 |
|   of which conflicting values | 134,954 |

## 3. Sampling intervals

Positive intervals: 7,500,055; mode = 7 s (24.1% of intervals); median = 7 s.

Gaps > 5 min: 452 (> 1 h: 33; > 1 day: 13). Ten largest:

| From | To | Hours |
|---|---|---|
| 2014-01-24 01:56:25 UTC | 2014-03-06 16:23:19 UTC | 998.45 |
| 2015-05-10 23:36:10 UTC | 2015-05-19 15:10:21 UTC | 207.57 |
| 2015-06-10 13:19:01 UTC | 2015-06-18 01:39:44 UTC | 180.35 |
| 2014-08-30 12:00:55 UTC | 2014-09-04 11:33:21 UTC | 119.54 |
| 2014-08-28 04:23:04 UTC | 2014-09-01 16:55:24 UTC | 108.54 |
| 2015-04-03 01:43:17 UTC | 2015-04-07 10:18:39 UTC | 104.59 |
| 2015-03-22 08:15:59 UTC | 2015-03-24 15:23:01 UTC | 55.12 |
| 2015-05-24 10:09:41 UTC | 2015-05-26 12:09:29 UTC | 50.00 |
| 2014-10-27 00:05:00 UTC | 2014-10-28 13:46:50 UTC | 37.70 |
| 2015-03-31 04:22:36 UTC | 2015-04-01 16:26:19 UTC | 36.06 |

Full interval histogram: `dt_histogram.csv`.

## 4. Per-channel raw value checks

| Channel | NaN | Non-numeric | Negative | > 25000 W | Zero | Min | Max |
|---|---|---|---|---|---|---|---|
| Aggregate | 188,973 | 0 | 0 | 3 | 12 | 0.0 | 29159.0 |
| Appliance1 | 437,890 | 0 | 0 | 96 | 6,144,617 | 0.0 | 76677.0 |
| Appliance2 | 459,468 | 0 | 0 | 15 | 5,186,394 | 0.0 | 98301.0 |
| Appliance3 | 311,961 | 0 | 0 | 14 | 4,867,346 | 0.0 | 73725.0 |
| Appliance4 | 510,963 | 0 | 0 | 177 | 7,971,461 | 0.0 | 98301.0 |
| Appliance5 | 497,223 | 0 | 0 | 13 | 7,843,948 | 0.0 | 73725.0 |
| Appliance6 | 342,337 | 0 | 0 | 1 | 8,100,901 | 0.0 | 37610.0 |
| Appliance7 | 232,752 | 0 | 0 | 22 | 7,398,037 | 0.0 | 98299.0 |
| Appliance8 | 390,705 | 0 | 0 | 29 | 6,589,179 | 0.0 | 98301.0 |
| Appliance9 | 1,911,811 | 0 | 0 | 13 | 1,616 | 0.0 | 65535.0 |

## 5. Cleaning actions (gap-fill limit = 5 min, cap = 25000 W)

Global sort needed: **True**. Duplicate-Unix rows dropped (keep first): **1,211,048**. Rows with missing Unix dropped: 0.

1-min span: 920,031 minutes; minutes with >=1 reading: 804,954.

| Channel | Neg->NaN | Over cap->NaN | Missing min (pre-fill) | Filled | Missing (post-fill) |
|---|---|---|---|---|---|
| Aggregate | 0 | 3 | 115137 | 853 | 114284 |
| Appliance1 | 0 | 84 | 117647 | 2126 | 115521 |
| Appliance2 | 0 | 13 | 118519 | 2132 | 116387 |
| Appliance3 | 0 | 13 | 116590 | 894 | 115696 |
| Appliance4 | 0 | 129 | 116009 | 942 | 115067 |
| Appliance5 | 0 | 12 | 123541 | 3123 | 120418 |
| Appliance6 | 0 | 1 | 115970 | 770 | 115200 |
| Appliance7 | 0 | 21 | 115125 | 850 | 114275 |
| Appliance8 | 0 | 29 | 116780 | 1591 | 115189 |
| Appliance9 | 0 | 8 | 212974 | 1304 | 211670 |

## 6. Long outages (Aggregate still missing after fill; never imputed)

Count: **376**; total missing: **114,284 min**.

Longest 15:

| Start UTC | End UTC | Minutes | Hours |
|---|---|---|---|
| 2014-01-24 01:57:00+00:00 | 2014-03-06 16:22:00+00:00 | 59906 | 998.43 |
| 2015-05-10 23:37:00+00:00 | 2015-05-19 15:09:00+00:00 | 12453 | 207.55 |
| 2015-06-10 13:20:00+00:00 | 2015-06-18 01:38:00+00:00 | 10819 | 180.32 |
| 2015-04-03 01:44:00+00:00 | 2015-04-07 10:17:00+00:00 | 6274 | 104.57 |
| 2015-03-22 08:16:00+00:00 | 2015-03-24 15:22:00+00:00 | 3307 | 55.12 |
| 2014-08-30 12:01:00+00:00 | 2014-09-01 16:54:00+00:00 | 3174 | 52.9 |
| 2015-05-24 10:10:00+00:00 | 2015-05-26 12:08:00+00:00 | 2999 | 49.98 |
| 2014-10-27 00:06:00+00:00 | 2014-10-28 13:45:00+00:00 | 2260 | 37.67 |
| 2015-03-31 04:23:00+00:00 | 2015-04-01 16:25:00+00:00 | 2163 | 36.05 |
| 2015-03-15 10:03:00+00:00 | 2015-03-16 16:15:00+00:00 | 1813 | 30.22 |
| 2013-11-07 10:04:00+00:00 | 2013-11-08 11:41:00+00:00 | 1538 | 25.63 |
| 2014-12-18 10:29:00+00:00 | 2014-12-19 10:36:00+00:00 | 1448 | 24.13 |
| 2014-11-10 00:18:00+00:00 | 2014-11-10 13:14:00+00:00 | 777 | 12.95 |
| 2014-07-07 03:48:00+00:00 | 2014-07-07 09:31:00+00:00 | 344 | 5.73 |
| 2015-01-29 15:20:00+00:00 | 2015-01-29 17:15:00+00:00 | 116 | 1.93 |

## 7. Flat-line check (Aggregate)

Runs of >= 120 identical consecutive 1-min values: **5**; longest = 1246 min. (Reported only, not removed.)
