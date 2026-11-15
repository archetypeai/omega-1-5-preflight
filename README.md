# omega-1-5-preflight

Predict whether a dataset is likely to reach **~70% accuracy** with Archetype AI's **Omega 1.5** embeddings and a KNN classifier — **before** building on it. Omega 1.5 is called through the Inference API (also called Model Serving; model `omega-embedding`).

This is the Omega 1.5 successor to [omega-1-4-preflight](https://github.com/archetypeai/omega-1-4-preflight), which targets `omega_1_4_base` through the `machine-state-classification` batch pipeline. The static checks carry over; the pilot is new. See [What changed from omega-1-4-preflight](#what-changed-from-omega-1-4-preflight).

## What it does

- **Static checks** (fast, local, no API): 12 checks across schema, timestamp monotonicity and gaps, missing values, constant columns, feature scale, n-shot support, cross-file schema match, class balance with majority baseline, window-vs-sampling physical time, an accuracy prior, window-size range (16–1024), and request size against the Inference API's 2 MiB limit. See [Static checks](#static-checks-whats-run).
- **Pilot run** (optional, needs API access): holds out the tail of each shot file as a labeled pilot slice, embeds the library and pilot windows with Omega 1.5 in one Inference API session, classifies the pilot windows with a local KNN, and reports a verdict against the 70% bar and the always-predict-majority baseline. Nothing is uploaded and no batch job is created.

v1 scope: **binary time-series** (normal vs fault).

## Setup

```bash
git clone https://github.com/archetypeai/omega-1-5-preflight.git
cd omega-1-5-preflight

cp .env.example .env
# Edit .env with your ATAI_API_KEY (production endpoint already set)

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The `.env` file wins over variables already exported in your shell, so an `ATAI_*` in a shell profile can't point the pilot at another deployment.

The pilot needs a production API key: the free trial (`try.archetypeai.app`) doesn't serve `omega-embedding`. The static checks need no key.

## Example data

Two ready-to-use shot files ship under `data/`:

| File | Rows | Class | Contents |
|---|---|---|---|
| `data/normal_shots.csv` | 2,000 | normal | Contiguous fault-free block (Tennessee Eastman Process) |
| `data/fault_shots.csv`  | 2,000 | fault  | Contiguous faulted block (Tennessee Eastman Process) |

Both are 53-column CSVs (timestamp + 52 process variables). See [Data attribution](#data-attribution).

## Usage

### Static checks only (no API calls)

```bash
.venv/bin/python preflight.py \
  --shots-normal data/normal_shots.csv \
  --shots-fault  data/fault_shots.csv
```

### With pilot run

```bash
.venv/bin/python preflight.py \
  --shots-normal data/normal_shots.csv \
  --shots-fault  data/fault_shots.csv \
  --pilot
```

### All flags

| Flag | Default | Meaning |
|---|---|---|
| `--shots-normal PATH` | required | CSV of contiguous normal-class rows |
| `--shots-fault PATH`  | required | CSV of contiguous fault-class rows |
| `--timestamp-column`  | `timestamp` | Timestamp column name |
| `--timestamp-unit U`  | `auto` | Physical unit of the timestamp column (`seconds`/`minutes`/`hours`), used to report the window span in natural language |
| `--window-size N`     | 64 | Rows per window (16–1024), set on the Inference API session |
| `--step-size N`       | `window_size // 4` | Rows between window starts |
| `--n-neighbors K`     | 5  | KNN neighbours |
| `--metric M`          | `euclidean` | Distance metric (`euclidean`/`cosine`/`manhattan`) |
| `--weights W`         | `uniform` | KNN weighting (`uniform`/`distance`) |
| `--workers N`         | 6 | Concurrent embed requests |
| `--nshot-floor N`     | 500 | Minimum usable rows per class |
| `--pilot`             | off | Run the held-out pilot |
| `--pilot-size N`      | auto | Override the auto split — rows per class for the pilot |
| `--force`             | off | Run the pilot even below the floor (underpowered) |
| `--env PATH`          | `.env` | Env file path |

## How the pilot works

1. **Split** each shot file: the first rows form the KNN library, the last rows the pilot slice (see [Pilot split rule](#pilot-split-rule)).
2. **Scale** every column with a mean and std fit on the library rows only, so the pilot never leaks into the scaler.
3. **Window** each contiguous slice into `window_size`-row windows every `step_size` rows. Library and pilot windows never mix classes.
4. **Embed** each window with Omega 1.5. One Inference API session is opened with `values: {"window_size": W, "step_size": W}`, and each request carries one window with all channels, returning one row of `channels × 768` values. If one window would exceed the 2 MiB request limit, its channels are split across requests; Omega 1.5 embeds each channel independently, so the result is identical. The session is cancelled when the pilot ends.
5. **Classify** each pilot window by a KNN vote over the library embeddings, and score against the known class.

### Calibration on the example data

Measured on production, 2026-11-15 (step = window ÷ 4, k=5, euclidean, uniform):

| Window | Windows (library / pilot) | Accuracy | Macro F1 | Verdict | Wall time |
|---|---|---|---|---|---|
| 32 | 394 / 94 | 0.926 | 0.925 | PASS | 13 min* |
| 64 (default) | 194 / 44 | 0.955 | 0.954 | PASS | 5 min 30 s |
| 128 | 94 / 18 | 1.000 | 1.000 | PASS | 3 min 45 s* |

The majority baseline is 0.500 throughout. At window 128 the pilot has only 9 windows per class, so its 1.000 is a small sample. *Ran while another job shared the model, so slower than alone.

These are within-distribution numbers and overstate accuracy on new data by a wide margin on this dataset: see [Pilot caveats](#pilot-caveats).

## Pilot split rule

| Rows per class | Split | Label |
|---|---|---|
| ≥ 1,000 | 80% shots / 20% pilot (pilot ≥ 300) | high-confidence |
| 500–999 | 70% shots / 30% pilot (pilot ≥ 300) | low-confidence |
| < 500   | refused (unless `--force`) | — |

## Static checks (what's run)

| # | Check | Type | Fails when |
|---|---|---|---|
| 1 | `schema` | universal | timestamp column missing or non-numeric cells present |
| 2 | `timestamp` | time-series | non-monotonic, or a gap larger than 5× the median delta. Omega windows are cut by sample index and never see timestamps, so a window spanning a gap would silently join two unrelated stretches |
| 3 | `missing_values` | universal | any column has missing values (warn; strongly warn if >5%) |
| 4 | `constant_columns` | universal | any feature column has zero variance (warn) |
| 5 | `feature_scale` | universal | info only — reports the range spread. Omega 1.5 z-scores each channel per window, and the pilot z-scores every column, so mixed scales don't distort the embeddings |
| 6 | `nshot_support` | labeled | fewer than `--nshot-floor` rows per class |
| 7 | `schema_match` | universal | shot files have different columns |
| 8 | `class_balance` | labeled | majority class > 70% (warn) or > 85% (strong warn) |
| 9 | `window_vs_sampling` | time-series | info only — reports the window span in rows or physical time |
| 10 | `accuracy_prior` | labeled | info only — recommends `--pilot` for a concrete number |
| 11 | `window_size_range` | universal | `--window-size` outside [16, 1024]. Shorter windows are padded and masked inside Omega 1.5; longer ones keep only their last 1024 samples |
| 12 | `request_size` | universal | info only — compares one window's request with the 2 MiB limit and says whether the pilot will split channels across requests |

## Pilot verdicts

| Verdict | Meaning |
|---|---|
| `PASS`  | Pilot accuracy ≥ 70% AND beats majority baseline by >5pp |
| `WARN`  | Pilot below 70% but at or above majority baseline — try another window size or metric |
| `FAIL`  | Pilot below majority baseline — Omega embeddings + KNN don't separate the classes at this window size |

## Pilot caveats

The held-out pilot measures **within-distribution** accuracy: the last ~20% of each shot file, drawn from the same runs, shifts or equipment as the library. That is easier than generalizing to new data, so a `PASS` is necessary but **not sufficient** evidence that the full dataset will clear 70%.

- **Measured with Omega 1.5 on TEP** (production, 2026-11-15, window 64): the bundled shot files as the library, scored on 300 windows from the TEP *Testing* split — simulation runs the shots never saw. 150 normal windows came from fault-free runs; 150 fault windows came from faulty runs across faults 1–20, each starting after the fault is introduced at sample 160.

  | | Pilot | New simulation runs |
  |---|---|---|
  | Normal recall | 1.000 | 0.993 |
  | Fault recall | 0.909 | **0.247** |
  | Accuracy | 0.955 (50/50 mix) | 0.620 balanced; **0.366** at TEP's natural mix of 16% normal / 84% fault, below the 0.840 always-predict-fault baseline |

  Fault recall by fault number: faults 1, 2 and 12 are caught every time and 8 and 13 three times in four, but faults 4, 9, 10, 11, 14, 15, 16, 19 and 20 are never caught. Reproduce with `scripts/tep_gap.py` (see [Reproducing the TEP gap](#reproducing-the-tep-gap)). `fault_shots.csv` is a single contiguous faulted block, so the library represents a narrow slice of fault behaviour; the pilot cannot reveal this because its held-out slice comes from the same block.
- **Observed with Omega 1.4 on TEP:** pilot 0.784 vs full inference 0.506–0.537 on 15.3M rows from different simulations. That run used a different, stitched three-fault library, so it is not directly comparable with the Omega 1.5 numbers above.
- **Be skeptical when** production data comes from runs, shifts, equipment, periods or conditions the shot files don't represent. The pilot cannot detect a shift it never sees.

Treat a `PASS` as "worth building on", not as "the full dataset will clear 70%". Treat a `WARN` or `FAIL` as strong evidence it won't without changes.

## What changed from omega-1-4-preflight

| | omega-1-4-preflight | omega-1-5-preflight |
|---|---|---|
| Model | `omega_1_4_base` | Omega 1.5 (`omega-embedding`) |
| Pilot | Uploads files and runs a `machine-state-classification` batch job (`/v0.5` API) | One Inference API session + local KNN; nothing uploaded |
| `--step-size` default | 1 | `window_size // 4` (each window is one embed call) |
| `timestamp` gap | warn | fail |
| `feature_scale` | warn above 3 orders of magnitude | info only |
| `request_size` check | — | new |
| `.env` precedence | shell wins | `.env` wins |

## Reproducing the TEP gap

`scripts/tep_gap.py` reruns the measurement in [Pilot caveats](#pilot-caveats). It needs the
Kaggle [TEP CSVs](https://www.kaggle.com/datasets/afrniomelo/tep-csv) (`TEP_FaultFree_Testing.csv`
and `TEP_Faulty_Testing.csv`) and pandas:

```bash
.venv/bin/pip install pandas
.venv/bin/python scripts/tep_gap.py --tep-dir ~/Downloads/kaggle-tep
```

It uses the same library, scaler, windowing and KNN as the pilot, samples test windows with a fixed
seed (`--seed`, default 42), and takes about 15 minutes on production at the defaults.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Static checks clean OR pilot verdict `PASS`/`WARN` |
| 1 | Static checks failed OR missing credentials |
| 2 | Pilot verdict `FAIL` |

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest
```

The tests need no API access: the pilot is exercised with a fake embed function.

## Licence

The code is under the [Apache License 2.0](LICENSE). The data is the Rieth et al.
release, dedicated to the public domain: see [Data attribution](#data-attribution).

## Data attribution

All process data used here comes from **Additional Tennessee Eastman Process
Simulation Data for Anomaly Detection Evaluation**:

- **Authors:** Cory A. Rieth, Ben D. Amsel, Randy Tran, Maia B. Cook.
- **Dataset:** Harvard Dataverse, 2017,
  [doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1).
- **Original process:** Downs, J. J., & Vogel, E. F. (1993). *A plant-wide
  industrial process control problem.* Computers & Chemical Engineering, 17(3),
  245–255.
- **Terms:** a public-domain dedication; the dataset's user agreement allows
  copying, modifying and distributing the work without permission.
- **No endorsement:** neither the authors nor Pacific Science & Engineering
  Group, Inc. are involved in this example or endorse it.

**What was changed** (the files in `data/`): each file is one contiguous 2,000-row block,
fault-free or faulted, extracted from a CSV conversion of the dataset
([afrniomelo/tep-csv](https://www.kaggle.com/datasets/afrniomelo/tep-csv) on Kaggle).
All 52 process variables are kept; the `timestamp` column is the integer sample index
from that pipeline.
