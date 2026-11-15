"""Pilot runner — holds out the tail of each shot file as a labeled pilot slice, embeds
library and pilot windows with Omega 1.5 over the Inference API, classifies the pilot windows
with a local KNN against the library, and scores them against their known class.

One Inference API session is opened with `window_size` / `step_size` set to the pilot's
window, so every request carries exactly one window (all channels) and returns one row of
`channels × 768` values. Nothing is uploaded and no batch job is created; the session is
cancelled when the pilot ends.
"""

from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import requests

from checks.static import BYTES_PER_VALUE, MAX_REQUEST_BYTES

MODEL = "omega-embedding"   # Omega 1.5 on the Inference API
EMBED_DIM = 768             # per channel
RETRIES = 4
RETRY_BACKOFF_SEC = 2.0  # doubles after each failed try: 2 s, 4 s, 8 s


@dataclass
class PilotConfig:
    window_size: int = 64
    n_neighbors: int = 5
    metric: str = "euclidean"
    weights: str = "uniform"
    step_size: Optional[int] = None   # None → window_size // 4
    timestamp_col: str = "timestamp"
    workers: int = 6

    @property
    def step(self) -> int:
        return self.step_size or max(1, self.window_size // 4)


@dataclass
class SplitPlan:
    shot_rows: int
    pilot_rows: int
    confidence: str  # "high" | "low" | "overridden" | "refused"
    reason: str


@dataclass
class PilotResult:
    verdict: str  # "PASS" | "WARN" | "FAIL"
    accuracy: float
    macro_f1: float
    per_class: dict
    confusion: dict
    majority_baseline: float
    confidence: str
    reason: str
    session_id: Optional[str] = None
    library_windows: int = 0
    pilot_windows: int = 0


# Minimum non-overlap guards for the --force and --pilot-size paths. Files at or
# above the 500-row "low confidence" threshold use the larger 100-row floors
# baked into those branches; these guards only kick in for smaller files where
# the old 100-row floors would have produced overlapping shots/pilot slices.
_MIN_SHOT_ROWS = 10
_MIN_PILOT_ROWS = 10


def plan_split(n_rows: int, force: bool = False, size_override: Optional[int] = None) -> SplitPlan:
    """Default 80/20 rule with floors, from the design spec."""
    if size_override is not None:
        pilot = min(size_override, max(0, n_rows - _MIN_SHOT_ROWS))
        shot = n_rows - pilot
        if (shot < _MIN_SHOT_ROWS or pilot < _MIN_PILOT_ROWS) and not force:
            return SplitPlan(0, 0, "refused",
                f"Override leaves shots={shot} pilot={pilot} "
                f"(need ≥{_MIN_SHOT_ROWS} each — pass --force to bypass).")
        return SplitPlan(shot, pilot, "overridden",
            f"User-specified pilot size {pilot} (shots {shot}).")

    if n_rows >= 1000:
        pilot = max(300, n_rows // 5)
        return SplitPlan(n_rows - pilot, pilot, "high",
            f"Standard 80/20 split ({n_rows - pilot} shots, {pilot} pilot).")
    if n_rows >= 500:
        pilot = max(300, n_rows * 3 // 10)
        shot = n_rows - pilot
        return SplitPlan(shot, pilot, "low",
            f"70/30 split near the floor ({shot} shots, {pilot} pilot). Low-confidence.")
    if force:
        pilot = max(_MIN_PILOT_ROWS, n_rows // 3)
        shot = n_rows - pilot
        if shot < _MIN_SHOT_ROWS:
            return SplitPlan(0, 0, "refused",
                f"Only {n_rows} rows — even with --force, a 1/3 pilot split "
                f"leaves shots={shot} (<{_MIN_SHOT_ROWS}). Expand shots or lower --pilot-size.")
        return SplitPlan(shot, pilot, "low",
            f"--force override below floor ({shot} shots, {pilot} pilot). Underpowered.")
    return SplitPlan(0, 0, "refused",
        f"Only {n_rows} rows (<500 floor). Use --pilot-set FILE, --force, or expand shots.")


# ---------------------------------------------------------------------------
# Inference API client
# ---------------------------------------------------------------------------
PROD_ENDPOINT = "https://platform.u1.archetypeai.app/api"
TRIAL_ENDPOINT = "https://try.archetypeai.app/api"
NAMED_ENDPOINTS = {"prod": PROD_ENDPOINT, "trial": TRIAL_ENDPOINT}


def choose_endpoint(flag: Optional[str], env_value: Optional[str]) -> str:
    """Pick the deployment: `--endpoint` (a name or a URL) wins, then ATAI_API_ENDPOINT,
    then production. Returns the resolved '<root>/api' URL."""
    raw = flag or env_value or "prod"
    return resolve_endpoint(NAMED_ENDPOINTS.get(raw.strip().lower(), raw))


def is_free_trial(endpoint: str) -> bool:
    """The free trial doesn't serve omega-embedding, so the pilot can't run there."""
    return "//try.archetypeai.app" in endpoint


def resolve_endpoint(raw: str) -> str:
    """Accept a bare root, a /vX.Y suffix, or /api itself; return '<root>/api'."""
    base = re.sub(r"/api(/lens)?$", "", (raw or "").rstrip("/"))
    base = re.sub(r"/v[0-9]+(\.[0-9]+)*$", "", base)
    if not base:
        raise ValueError("ATAI_API_ENDPOINT is not set")
    return base + "/api"


class ServingClient:
    """Omega 1.5 over the Inference API: one session, one request per window."""

    def __init__(self, api_key: str, endpoint: str, timeout: int = 70):
        self.base = resolve_endpoint(endpoint)
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._values: dict = {}

    def _post(self, path: str, body: Optional[dict] = None) -> dict:
        r = requests.post(f"{self.base}{path}", headers=self.headers,
                          data=json.dumps(body) if body is not None else None, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} on {path}: {r.text[:200]}")
        return r.json() if r.content else {}

    def open_session(self, window_size: int) -> str:
        self._values = {"window_size": window_size, "step_size": window_size}
        self.session_id = self._post("/inference/sessions", {"model": MODEL, "values": self._values})["id"]
        return self.session_id

    def close(self) -> None:
        if self.session_id:
            try:
                self._post(f"/inference/sessions/{self.session_id}/cancel")
            except Exception:  # noqa: BLE001 — an idle session expires on its own
                pass

    def _embed_group(self, channels: list) -> list:
        body = {"model": MODEL, "query": "",
                "events": [{"type": "data.numeric_array", "event_data": {"contents": channels}}]}
        last: Optional[Exception] = None
        for attempt in range(RETRIES):
            if attempt:
                time.sleep(RETRY_BACKOFF_SEC * 2 ** (attempt - 1))
            try:
                out = self._post(f"/inference/sessions/{self.session_id}/query", body)
                resp = out.get("response") or {}
                if not resp.get("success"):
                    raise RuntimeError(f"success=false: {str(resp)[:120]}")
                rows = resp["response"]
                if len(rows) != 1 or len(rows[0]) != EMBED_DIM * len(channels):
                    raise RuntimeError(f"unexpected shape {len(rows)} x {len(rows[0]) if rows else 0}")
                return rows[0]
            except requests.RequestException as exc:
                # A dropped or reset connection, or a timeout: the request may never
                # have reached the model, so sending it again is safe.
                last = exc
            except RuntimeError as exc:
                last = exc
                if "session_not_running" in str(exc):
                    self.open_session(self._values["window_size"])
        raise RuntimeError(f"embed failed after {RETRIES} tries: {last}")

    def embed_window(self, channels: list) -> list:
        """Embed one [channels x window] window. Channels are split across requests when
        one request would exceed the 2 MiB body limit; Omega 1.5 embeds each channel
        independently, so the concatenated result is identical."""
        per_request = max(1, int(MAX_REQUEST_BYTES * 0.9) // (len(channels[0]) * BYTES_PER_VALUE))
        out: list = []
        for i in range(0, len(channels), per_request):
            out.extend(self._embed_group(channels[i:i + per_request]))
        return out


# ---------------------------------------------------------------------------
# Data prep: read, split, scale, window
# ---------------------------------------------------------------------------
def read_matrix(path: str, data_columns: list) -> np.ndarray:
    """Read the feature columns of a shot CSV into a (rows, channels) float array."""
    with open(path, newline="") as f:
        rows = [[float(r[c]) for c in data_columns] for r in csv.DictReader(f)]
    return np.asarray(rows, dtype=np.float64)


def fit_scaler(blocks: list) -> tuple:
    """Per-channel mean/std from the shot (library) rows only — the pilot never leaks in."""
    pool = np.concatenate(blocks, axis=0)
    std = pool.std(axis=0)
    return pool.mean(axis=0), np.where(std < 1e-12, 1.0, std)


def windows(block: np.ndarray, window: int, step: int) -> list:
    """Channel-first [channels x window] windows over one contiguous block."""
    return [block[s:s + window].T.tolist() for s in range(0, len(block) - window + 1, step)]


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------
def knn_predict(lib_X: np.ndarray, lib_y: list, X: np.ndarray, k: int,
                metric: str = "euclidean", weights: str = "uniform") -> list:
    if metric == "cosine":
        a = lib_X / np.maximum(np.linalg.norm(lib_X, axis=1, keepdims=True), 1e-12)
        b = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        d = 1.0 - b @ a.T
    elif metric == "manhattan":
        d = np.abs(X[:, None, :] - lib_X[None, :, :]).sum(axis=-1)
    else:
        d = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] + (lib_X ** 2).sum(1)[None, :]
                               - 2 * X @ lib_X.T, 0.0))
    preds = []
    for row in d:
        votes: dict = {}
        for j in np.argsort(row)[:k]:
            w = 1.0 / max(row[j], 1e-12) if weights == "distance" else 1.0
            votes[lib_y[j]] = votes.get(lib_y[j], 0.0) + w
        preds.append(max(votes, key=votes.get))
    return preds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(predictions: dict, labels: dict, classes: list = None) -> dict:
    classes = classes or ["normal", "fault"]
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    correct = matched = 0
    for ts, pred in predictions.items():
        if ts not in labels:
            continue
        matched += 1
        actual = labels[ts]
        ai, pi = idx.get(actual), idx.get(pred)
        if ai is not None and pi is not None:
            cm[ai][pi] += 1
            if actual == pred:
                correct += 1

    accuracy = correct / matched if matched else 0.0
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        sup = sum(cm[i])
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": sup}
    macro_f1 = sum(pc["f1"] for pc in per_class.values()) / n
    total_support = sum(pc["support"] for pc in per_class.values())
    majority_support = max(pc["support"] for pc in per_class.values())
    majority_baseline = majority_support / total_support if total_support else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": {classes[i]: {classes[j]: cm[i][j] for j in range(n)} for i in range(n)},
        "matched": matched,
        "majority_baseline": majority_baseline,
    }


def verdict_for(m: dict) -> tuple:
    thresh = 0.70
    if m["accuracy"] >= thresh and m["accuracy"] > m["majority_baseline"] + 0.05:
        return "PASS", (f"Accuracy {m['accuracy']:.3f} clears 70% bar and beats majority "
                        f"baseline {m['majority_baseline']:.3f}.")
    if m["accuracy"] >= m["majority_baseline"]:
        tie = "ties" if abs(m["accuracy"] - m["majority_baseline"]) < 0.01 else "above"
        return "WARN", (f"Accuracy {m['accuracy']:.3f} below 70% bar (but {tie} majority baseline "
                        f"{m['majority_baseline']:.3f}). Try another --window-size or --metric.")
    return "FAIL", (f"Accuracy {m['accuracy']:.3f} < majority baseline {m['majority_baseline']:.3f}. "
                    f"Omega embeddings + KNN don't separate these classes at this window size; try "
                    f"another window, or fit a managed agent on the data.")


def run_pilot(
    normal_path: str,
    fault_path: str,
    normal_rows: int,
    fault_rows: int,
    data_columns: list,
    cfg: PilotConfig,
    client: ServingClient,
    force: bool = False,
    pilot_size: Optional[int] = None,
    embed: Optional[Callable[[list], list]] = None,
) -> PilotResult:
    """`embed` overrides the per-window embed call (tests pass a fake one)."""
    n_plan = plan_split(normal_rows, force=force, size_override=pilot_size)
    f_plan = plan_split(fault_rows, force=force, size_override=pilot_size)
    if n_plan.confidence == "refused" or f_plan.confidence == "refused":
        return PilotResult(
            verdict="FAIL", accuracy=0.0, macro_f1=0.0, per_class={}, confusion={},
            majority_baseline=0.0, confidence="refused",
            reason=f"Split refused. normal: {n_plan.reason} | fault: {f_plan.reason}",
        )
    print(f"[pilot] split plan -> normal: {n_plan.reason}")
    print(f"[pilot] split plan -> fault:  {f_plan.reason}")

    normal = read_matrix(normal_path, data_columns)
    fault = read_matrix(fault_path, data_columns)
    parts = {
        "normal": (normal[:n_plan.shot_rows], normal[n_plan.shot_rows:n_plan.shot_rows + n_plan.pilot_rows]),
        "fault": (fault[:f_plan.shot_rows], fault[f_plan.shot_rows:f_plan.shot_rows + f_plan.pilot_rows]),
    }
    mean, std = fit_scaler([shot for shot, _ in parts.values()])
    w, step = cfg.window_size, cfg.step

    lib_w, lib_y, pil_w, pil_y = [], [], [], []
    for label, (shot, pilot) in parts.items():
        lw = windows((shot - mean) / std, w, step)
        pw = windows((pilot - mean) / std, w, step)
        lib_w += lw; lib_y += [label] * len(lw)
        pil_w += pw; pil_y += [label] * len(pw)
    print(f"[pilot] {len(lib_w)} library + {len(pil_w)} pilot windows "
          f"(window={w}, step={step}, {len(data_columns)} channels)")
    if not lib_w or not pil_w:
        return PilotResult(
            verdict="FAIL", accuracy=0.0, macro_f1=0.0, per_class={}, confusion={},
            majority_baseline=0.0, confidence=n_plan.confidence,
            reason=f"No complete windows of {w} rows in the library or pilot slices.",
        )

    session_id = None
    if embed is None:
        session_id = client.open_session(w)
        print(f"[pilot] opened Inference API session {session_id} (window_size={w})")
        embed = client.embed_window
    try:
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            lib_X = np.asarray(list(ex.map(embed, lib_w)))
            pil_X = np.asarray(list(ex.map(embed, pil_w)))
    finally:
        if session_id:
            client.close()
            print("[pilot] session cancelled")

    preds = knn_predict(lib_X, lib_y, pil_X, cfg.n_neighbors, cfg.metric, cfg.weights)
    m = score(dict(enumerate(preds)), dict(enumerate(pil_y)))
    verdict, reason = verdict_for(m)
    confidence = "low" if "low" in (n_plan.confidence, f_plan.confidence) else n_plan.confidence
    return PilotResult(
        verdict=verdict, accuracy=m["accuracy"], macro_f1=m["macro_f1"],
        per_class=m["per_class"], confusion=m["confusion"],
        majority_baseline=m["majority_baseline"], confidence=confidence, reason=reason,
        session_id=session_id, library_windows=len(lib_w), pilot_windows=len(pil_w),
    )
