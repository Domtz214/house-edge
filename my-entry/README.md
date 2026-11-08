# House Edge entry — v2 (multinomial logistic regression)

## Entry
- Team ID: [FILL IN TEAM ID]
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submission.csv
- Forecast SHA-256: 52b534a0916cfa049eef723035c466d211ab3b9dc193333a3d7b37ecf7776858
- Environment: `python` (Python 3.13, standard library only)

## Setup
- Runtime: Python 3.11+ (tested 3.13), Linux, CPU only, < 1 GB RAM
- Dependencies: Python standard library only (inference). Training (not needed to reproduce) used numpy + scikit-learn.
- Data: the round's `public/` folder, passed via `--data`

## Recreate the forecast
Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

- Loads the saved weights in `model.json` (no training at run time); fully deterministic, no randomness.
- Features (`features.py`): per-machine time-decayed payout frequencies (half-lives 4–64 ticks + all history),
  same-episode other-machine pooled frequencies (half-lives 8, 32), one-hot payouts at lags 1–8
  (missing ticks flagged, never treated as zero), short-vs-long drift terms, signal one-hot.
  All features use only ticks < cutoff. Features are interacted with machine type.
- Model: L2 multinomial logistic regression, C chosen on validation, refit on train + validation.
- Training command (separate, local only): `python train.py --data public --entry my-entry` (seed 0).
- Measured runtime: ~10 seconds on a laptop CPU.

## Local check
- `check_submission.py` result: valid, 1536 test queries.
- Validation (train-only model): 30.84 / 60 technical points vs 30.00 for the pooled reference.
