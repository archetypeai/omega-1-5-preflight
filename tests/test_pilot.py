import numpy as np
import pytest

from checks import pilot
from checks.pilot import (PilotConfig, ServingClient, knn_predict, plan_split, resolve_endpoint,
                          run_pilot, score, windows)
from checks.static import summarize_shot_file


def test_plan_split_rules():
    assert (plan_split(2000).shot_rows, plan_split(2000).pilot_rows) == (1600, 400)
    assert plan_split(700).confidence == "low"
    assert plan_split(400).confidence == "refused"
    assert plan_split(400, force=True).confidence == "low"


def test_score_counts_and_baseline():
    m = score({0: "normal", 1: "fault", 2: "fault", 3: "fault"},
              {0: "normal", 1: "fault", 2: "normal", 3: "fault"})
    assert m["accuracy"] == 0.75
    assert m["majority_baseline"] == 0.5
    assert m["confusion"]["normal"]["fault"] == 1


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_separates_clusters(metric, weights):
    rng = np.random.default_rng(0)
    lib = np.vstack([rng.normal([1, 0], 0.05, (10, 2)), rng.normal([0, 1], 0.05, (10, 2))])
    y = ["a"] * 10 + ["b"] * 10
    X = np.array([[0.9, 0.1], [0.1, 0.9]])
    assert knn_predict(lib, y, X, 5, metric, weights) == ["a", "b"]


def test_windows_are_channel_first():
    block = np.arange(20, dtype=float).reshape(10, 2)   # 10 rows, 2 channels
    w = windows(block, 4, 2)
    assert len(w) == 4 and len(w[0]) == 2 and len(w[0][0]) == 4
    assert w[1][0] == [4.0, 6.0, 8.0, 10.0]


@pytest.mark.parametrize("raw", ["https://x.example", "https://x.example/", "https://x.example/api",
                                 "https://x.example/api/lens", "https://x.example/v0.5"])
def test_resolve_endpoint(raw):
    assert resolve_endpoint(raw) == "https://x.example/api"


def test_embed_window_splits_channels_over_the_limit(monkeypatch):
    calls = []
    c = ServingClient("k", "https://x.example")

    def fake_group(channels):
        calls.append(len(channels))
        return [0.0] * (768 * len(channels))

    monkeypatch.setattr(c, "_embed_group", fake_group)
    out = c.embed_window([[0.0] * 1024 for _ in range(200)])
    assert len(out) == 200 * 768 and len(calls) > 1 and sum(calls) == 200


def test_run_pilot_end_to_end_with_fake_embed(shot_files):
    normal, fault = shot_files
    cols = summarize_shot_file(normal).feature_cols

    def fake_embed(window):   # per-channel mean and std, enough to separate the classes
        return [v for ch in window for v in (float(np.mean(ch)), float(np.std(ch)))]

    res = run_pilot(normal, fault, 1200, 1200, cols, PilotConfig(window_size=32),
                    client=None, embed=fake_embed)
    assert res.verdict == "PASS" and res.accuracy == 1.0
    assert res.library_windows > 0 and res.pilot_windows > 0
    assert res.session_id is None


def test_run_pilot_refuses_tiny_files(shot_files):
    res = run_pilot(*shot_files, 300, 300, ["c0"], PilotConfig(), client=None, embed=lambda w: [0.0])
    assert res.verdict == "FAIL" and res.confidence == "refused"


def test_embed_group_retries_dropped_connections(monkeypatch):
    """A reset connection or timeout is retried (with backoff) instead of ending the pilot."""
    import requests
    import checks.pilot as pilot
    monkeypatch.setattr(pilot.time, "sleep", lambda s: None)
    c = ServingClient("k", "https://x.example")
    c.session_id = "ses_x"
    calls = []

    def flaky_post(path, body=None):
        calls.append(path)
        if len(calls) == 1:
            raise requests.exceptions.ChunkedEncodingError("Connection broken: reset by peer")
        if len(calls) == 2:
            raise requests.exceptions.ConnectionError("connection refused")
        return {"response": {"success": True, "response": [[0.0] * pilot.EMBED_DIM]}}

    monkeypatch.setattr(c, "_post", flaky_post)
    assert len(c._embed_group([[0.0] * 16])) == pilot.EMBED_DIM
    assert len(calls) == 3


def test_embed_group_gives_up_after_retries(monkeypatch):
    import requests
    import pytest
    import checks.pilot as pilot
    monkeypatch.setattr(pilot.time, "sleep", lambda s: None)
    c = ServingClient("k", "https://x.example")
    c.session_id = "ses_x"

    def always_down(path, body=None):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr(c, "_post", always_down)
    with pytest.raises(RuntimeError, match="after 4 tries"):
        c._embed_group([[0.0] * 16])
