# House Edge ADAPTATION entry — v6-adapt (lag-pair pattern features + surprise features + LightGBM, stdlib inference)

## Entry
- Team name: ColorWICS
- Round: adaptation (round two)
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submission.csv
- Forecast SHA-256: 52d503ed2eab0eb6ceb8a759ad3b6b5ecc7518532b49fa4ca85ac320bb8486b9
- Code ZIP: entry.zip
- Environment: `python` (Python 3.13, standard library only at prediction time)

## Setup
- Runtime: Python 3.11+ (tested 3.13), Linux, CPU only, < 1 GB RAM
- Prediction dependencies: Python standard library only.
- Dependencies: `requirements.txt` (empty - standard library only for prediction).
- Training dependencies (not needed to reproduce the CSV): `training/requirements-train.txt`
  (numpy 2.5.3, lightgbm 4.7.0; LightGBM deterministic=True, random_state=0).
- Data: the round's `public/` folder, passed via `--data`.

## Recreate the forecast
Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

- Loads `win_model.json` and `size_model.json` (LightGBM trees exported with `dump_model()`),
  evaluated in pure Python by `features.py`. No training at run time, no randomness.
- Measured runtime: well under a minute on one CPU core.
- Training command (separate, optional; rewrites the two model JSON files deterministically):
  `pip install -r training/requirements-train.txt && python training/train_v6.py --data /path/to/public --entry . --report /tmp/report --final`

## Method
- Adaptation change: the recurring lag pairs changed. Re-discovered on the adaptation TRAIN split
  (most-informative pair per machine over its last 30 ticks): (1,2), (1,4), (2,6), (3,8)
  (main round used (1,2), (1,3), (2,5), (4,7)). The two payout-size profiles per type were
  unchanged. Models were retrained on adaptation data only; main-round data (/main_data) is not used.
- Win chance: for each machine and each candidate lag pair (1,2), (1,4), (2,6), (3,8), a
  recency-weighted table of win/lose given the win/lose outcomes at the two lags (half-lives
  6/12/25/60 ticks, Beta(0.1,0.1) prior), plus Bayesian evidence weights for each pair. A LightGBM
  classifier, trained on pseudo-queries (every observed tick, using only earlier ticks), maps these
  to P(win).
- Payout size given a win: recency-weighted size frequencies and the log-likelihood ratio between
  each machine type's two size profiles, mapped by a LightGBM 4-class model.
- Change detection: recency-weighted log-loss (half-life 8) of the short- and long-memory
  pattern predictions for the machine itself and averaged over the other machines in the episode
  (earlier ticks only), so the model can react to drift and episode-wide shared changes.
- Final probability: [1 - P(win), P(win) * P(size | win)].
- All features use only ticks < cutoff; missing ticks are unknown, never zero.

## Local check
- Validation (models trained on adaptation train-split histories only): 18.03 / 20 technical points
  (unchanged main-round model: 9.85 / 20). Final models are refit on adaptation train + validation histories.
