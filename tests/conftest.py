import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_csv(path, n_rows, n_cols=3, start=0, step=1.0, gap_at=None, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp"] + [f"c{i}" for i in range(n_cols)])
        ts = float(start)
        for r in range(n_rows):
            if gap_at is not None and r == gap_at:
                ts += 100 * step
            w.writerow([ts] + [offset + np.sin(r / (5 + i)) + 0.05 * rng.standard_normal() for i in range(n_cols)])
            ts += step
    return str(path)


@pytest.fixture
def shot_files(tmp_path):
    normal = write_csv(tmp_path / "normal.csv", 1200, offset=0.0, seed=1)
    fault = write_csv(tmp_path / "fault.csv", 1200, offset=3.0, seed=2)
    return normal, fault
