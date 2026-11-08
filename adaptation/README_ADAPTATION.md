# ADAPTATION ROUND (round two) — ColorWICS

Everything for round two lives in this `adaptation/` folder. The main-round submission
(`SUBMIT_MAIN_FINAL/`, `my-entry-v6/`) is unchanged.

| Path | What |
| --- | --- |
| `SUBMIT_ADAPTATION/submission.csv` | Adaptation test forecast to upload |
| `SUBMIT_ADAPTATION/entry.zip` | Matching adaptation code ZIP |
| `entry/` | Source of the adaptation entry (v6 with re-discovered lag pairs, retrained) |
| `kit/` | Organizer adaptation kit (data + checker) |
| `report-trainonly/` | Validation predictions from the train-only model (18.03 / 20) |

Change vs main: lag pairs (1,2),(1,3),(2,5),(4,7) -> (1,2),(1,4),(2,6),(3,8), found on the adaptation train split.
