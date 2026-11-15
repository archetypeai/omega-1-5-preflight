#!/usr/bin/env python3
"""Pilot-vs-new-runs gap on TEP: how well does the bundled library generalize?

The library is the bundled shot files (data/normal_shots.csv, data/fault_shots.csv),
scaled with their own statistics, exactly as the pilot does. The test windows come from
the Kaggle TEP *Testing* split, whose simulation runs the shot files never saw:

  normal - random windows from random fault-free runs
  fault  - random windows from random faulty runs (faults 1-20), each starting after the
           fault is introduced at sample 160

Reports recall per class, recall per fault number, balanced accuracy, and accuracy at TEP's
natural class mix (16% normal / 84% fault), the figure the README's Pilot caveats cite.

Needs TEP_FaultFree_Testing.csv and TEP_Faulty_Testing.csv from
https://www.kaggle.com/datasets/afrniomelo/tep-csv, plus pandas:

    .venv/bin/pip install pandas
    .venv/bin/python scripts/tep_gap.py --tep-dir ~/Downloads/kaggle-tep
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from checks.pilot import (ServingClient, choose_endpoint, fit_scaler, is_free_trial,  # noqa: E402
                          knn_predict, read_matrix, windows)
from checks.static import summarize_shot_file  # noqa: E402

FAULT_ONSET = 160      # Testing split: fault introduced at sample 160
RUN_LENGTH = 960       # samples per Testing simulation run
NATURAL_NORMAL = 0.16  # share of normal rows across the labeled TEP data


def load_env(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()


def load_windows(path, picks, cols, window, mean, std):
    """Read only the (fault, run) pairs that were picked, then cut each window."""
    want = {(f, r) for f, r, _ in picks}
    keep = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        mask = [(f, r) in want for f, r in zip(chunk["faultNumber"], chunk["simulationRun"])]
        if any(mask):
            keep.append(chunk[mask])
    runs = {k: g.sort_values("sample") for k, g in pd.concat(keep).groupby(["faultNumber", "simulationRun"])}
    out = []
    for f, r, s in picks:
        g = runs[(f, r)]
        block = g[(g["sample"] >= s) & (g["sample"] < s + window)][cols].to_numpy(float)
        if len(block) != window:
            raise ValueError(f"fault {f} run {r} start {s}: got {len(block)} rows")
        out.append((f, ((block - mean) / std).T.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tep-dir", required=True, help="Folder with the Kaggle TEP Testing CSVs")
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--per-class", type=int, default=150, help="Test windows per class")
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--env", default=os.path.join(REPO, ".env"))
    ap.add_argument("--endpoint", default=None, help="prod (default), trial, or a full URL; overrides ATAI_API_ENDPOINT")
    a = ap.parse_args()
    load_env(a.env)
    rng = random.Random(a.seed)
    w, step = a.window_size, max(1, a.window_size // 4)

    cols = summarize_shot_file(os.path.join(REPO, "data/normal_shots.csv")).feature_cols
    normal, fault = (read_matrix(os.path.join(REPO, f"data/{n}_shots.csv"), cols) for n in ("normal", "fault"))
    mean, std = fit_scaler([normal, fault])
    lib_n, lib_f = windows((normal - mean) / std, w, step), windows((fault - mean) / std, w, step)
    lib_w, lib_y = lib_n + lib_f, ["normal"] * len(lib_n) + ["fault"] * len(lib_f)

    normal_picks = [(0, rng.randint(1, 500), rng.randint(1, RUN_LENGTH - w + 1)) for _ in range(a.per_class)]
    fault_picks = [(rng.randint(1, 20), rng.randint(1, 500), rng.randint(FAULT_ONSET + 1, RUN_LENGTH - w + 1))
                   for _ in range(a.per_class)]
    t = time.time()
    test = (load_windows(os.path.join(a.tep_dir, "TEP_FaultFree_Testing.csv"), normal_picks, cols, w, mean, std)
            + load_windows(os.path.join(a.tep_dir, "TEP_Faulty_Testing.csv"), fault_picks, cols, w, mean, std))
    print(f"window={w} step={step}: {len(lib_w)} library windows, {len(test)} test windows")

    endpoint = choose_endpoint(a.endpoint, os.environ.get("ATAI_API_ENDPOINT"))
    if is_free_trial(endpoint):
        sys.exit(f"The free trial ({endpoint}) doesn't serve omega-embedding; use a production endpoint.")
    print(f"endpoint: {endpoint}")
    client = ServingClient(os.environ["ATAI_API_KEY"], endpoint)
    client.open_session(w)
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            lib_X = np.asarray(list(ex.map(client.embed_window, lib_w)))
            test_X = np.asarray(list(ex.map(client.embed_window, [x for _, x in test])))
    finally:
        client.close()

    pred = knn_predict(lib_X, lib_y, test_X, a.n_neighbors)
    true = ["normal" if f == 0 else "fault" for f, _ in test]
    rec = {c: float(np.mean([p == c for p, y in zip(pred, true) if y == c])) for c in ("normal", "fault")}
    per_fault = defaultdict(list)
    for (f, _), p in zip(test, pred):
        if f:
            per_fault[f].append(p == "fault")
    natural = NATURAL_NORMAL * rec["normal"] + (1 - NATURAL_NORMAL) * rec["fault"]
    print(f"recall normal={rec['normal']:.3f} fault={rec['fault']:.3f} | "
          f"balanced accuracy={(rec['normal'] + rec['fault']) / 2:.3f} | "
          f"natural-mix accuracy={natural:.3f} (always-predict-fault baseline {1 - NATURAL_NORMAL:.3f}) | "
          f"{time.time() - t:.0f}s")
    print("fault recall by fault number:", {f: round(float(np.mean(v)), 2) for f, v in sorted(per_fault.items())})


if __name__ == "__main__":
    main()
