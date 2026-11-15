from checks.static import (FAIL, INFO, PASS, check_feature_scale, check_request_size,
                           check_timestamp, check_window_size_range, run_all_static_checks,
                           summarize_shot_file)
from tests.conftest import write_csv


def test_clean_files_have_no_failures(shot_files):
    results, normal, fault = run_all_static_checks(*shot_files, window_size=64)
    assert not [r for r in results if r.level == FAIL]
    assert normal.feature_cols == ["c0", "c1", "c2"]
    assert {r.name for r in results} >= {"request_size", "window_size_range", "feature_scale"}


def test_timestamp_gap_fails(tmp_path):
    s = summarize_shot_file(write_csv(tmp_path / "gap.csv", 600, gap_at=300))
    r = check_timestamp(s)
    assert r.level == FAIL and "sample index" in r.message


def test_feature_scale_is_info_only(tmp_path):
    p = tmp_path / "scales.csv"
    with open(p, "w") as f:
        f.write("timestamp,small,big\n")
        for i in range(50):
            f.write(f"{i},{i * 1e-4},{i * 1e4}\n")
    assert check_feature_scale(summarize_shot_file(str(p))).level == INFO


def test_window_size_range_edges():
    assert check_window_size_range(15).level == FAIL
    assert check_window_size_range(16).level == PASS
    assert check_window_size_range(1024).level == PASS
    assert check_window_size_range(1025).level == FAIL


def test_request_size_mentions_split_only_when_needed():
    small = check_request_size(52, 64)
    big = check_request_size(200, 1024)
    assert small.level == big.level == INFO
    assert "fits in one request" in small.message
    assert "splits channels" in big.message and big.details["channels_per_request"] < 200
