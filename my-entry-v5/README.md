# House Edge entry — v5 (lag-pair pattern features + LightGBM, stdlib inference)

## Entry
- Team ID: [FILL IN TEAM ID]
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submission.csv
- Environment: `python` (Python 3.13, standard library only at prediction time)

## Setup
- Runtime: Python 3.11+ (tested 3.13), Linux, CPU only, < 1 GB RAM
- Prediction dependencies: Python standard library only.
- Training dependencies (not needed to reproduce): numpy, lightgbm 4.x (deterministic=True, seed 0).
- Data: the round's `public/` folder, passed via `--data`.

## Recreate the forecast
Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

- Loads `win_model.json` and `size_model.json` (LightGBM trees exported with `dump_model()`),
  evaluated in pure Python by `features.py`. No training at run time, no randomness.
- Measured runtime: well under a minute on one CPU core.
- Training command (separate, local): `python research/train_v5.py --data public --entry my-entry-v5`.

## Method
- Win chance: for each machine and each candidate lag pair (1,2), (1,3), (2,5), (4,7), a
  recency-weighted table of win/lose given the win/lose outcomes at the two lags (half-lives
  6/12/25/60 ticks, Beta(0.1,0.1) prior), plus Bayesian evidence weights for each pair. A LightGBM
  classifier, trained on pseudo-queries (every observed tick, using only earlier ticks), maps these
  to P(win).
- Payout size given a win: recency-weighted size frequencies and the log-likelihood ratio between
  each machine type's two size profiles, mapped by a LightGBM 4-class model.
- Final probability: [1 - P(win), P(win) * P(size | win)].
- All features use only ticks < cutoff; missing ticks are unknown, never zero.

## Local check
- Validation (models trained on train-split histories only): 53.82 / 60 technical points.
