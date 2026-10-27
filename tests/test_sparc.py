"""SPARC shares the tracking windows and library calculation at every entry point."""

from types import SimpleNamespace

import numpy as np
import pytest
from matchmaker.utils.eval import compute_sparc, evaluate_alignment, onset_wise_windows

from matchmaker_eval.eval import run_evaluation
from matchmaker_eval.leaderboard import metric_columns
from matchmaker_eval.utils import compute_event_pooled_summary
from matchmaker_eval.verify_tracking import _onset_wise_window_profile


def test_tracking_and_sparc_use_identical_windows():
    onsets = np.array([0.0, 0.0, 1.7, 8.0, 17.0, 28.0, 31.0, 44.0])
    errors = np.arange(len(onsets), dtype=float) / 10
    windows = onset_wise_windows(onsets)
    tracking = _onset_wise_window_profile(onsets, errors, 30.0, 1.0)
    times = np.arange(0.0, 44.02, 0.02)
    sparc = compute_sparc(times, times, onset_times=onsets, return_details=True)
    assert [(w["t0"], w["t1"]) for w in tracking] == [
        (w["t0"], w["t1"]) for w in sparc["sparc_windows"]
    ]
    assert tracking[-1]["t1"] == 44.0
    for window, result in zip(windows, tracking):
        start, stop = window["onset_index_start"], window["onset_index_stop"]
        assert result["deviation"] == np.median(errors[start:stop])
        assert result["n_points"] == stop - start


def test_evaluator_and_pooled_summary_match_library(tmp_path):
    times = np.arange(0.0, 12.02, 0.02)
    beats = times + 0.01 * np.sin(2 * np.pi * times)
    gt_times = np.arange(0.0, 13.0)
    path = np.array([times, beats])
    follower = SimpleNamespace(alignment_path=path)
    mm = SimpleNamespace(
        _has_run=True,
        score_follower=follower,
        score_part=SimpleNamespace(note_array=lambda: np.array([])),
        _wp_perf_to_seconds=lambda values: values,
        alignment_duration=None,
        input_type="midi",
    )
    expected = evaluate_alignment(beats, times, gt_times, gt_times)
    actual = run_evaluation(mm, gt=np.column_stack([gt_times, gt_times]))
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], nan_ok=True)
    assert actual.keys() == expected.keys()

    np.savetxt(
        tmp_path / "wp_0.tsv",
        path.T,
        delimiter="\t",
        header="perf_sec\tscore_beat",
        comments="",
    )
    np.savetxt(
        tmp_path / "gt_0.tsv",
        np.column_stack([gt_times, gt_times]),
        delimiter="\t",
        header="perf_sec\tscore_beat",
        comments="",
    )
    summary = compute_event_pooled_summary({"Index": [0], "tracked": [True]}, tmp_path)
    assert summary["smoothness"] == expected["smoothness"]
    columns = metric_columns({"summary_tracked": summary})
    assert columns["smoothness"] == expected["smoothness"]
    for key in expected:
        if key.startswith("accuracy_"):
            assert columns[key] == expected[key]


def test_pooling_skips_undefined_smoothness(tmp_path):
    onsets = np.arange(0.0, 61.0)
    for index in range(2):
        times = np.arange(0.0, 60.02, 0.02)
        beats = times if index == 0 else np.zeros_like(times)
        np.savetxt(
            tmp_path / f"wp_{index}.tsv",
            np.column_stack([times, beats]),
            delimiter="\t",
            header="perf_sec\tscore_beat",
            comments="",
        )
        np.savetxt(
            tmp_path / f"gt_{index}.tsv",
            np.column_stack([onsets, onsets]),
            delimiter="\t",
            header="perf_sec\tscore_beat",
            comments="",
        )
    summary = compute_event_pooled_summary(
        {"Index": [0, 1], "tracked": [True, True]}, tmp_path
    )
    assert summary["smoothness"] is not None
    assert not any(key.startswith("sparc") for key in summary)
    frozen = compute_event_pooled_summary({"Index": [1], "tracked": [True]}, tmp_path)
    assert frozen["smoothness"] is None
