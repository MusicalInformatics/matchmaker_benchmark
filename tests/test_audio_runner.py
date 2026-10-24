"""Piece selection, parallel dispatch and persistence after a failed piece."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from matchmaker_eval import test_audio as runner


@pytest.fixture
def pieces(tmp_path, monkeypatch):
    score = tmp_path / "score.musicxml"
    score.touch()
    metadata = pd.DataFrame(
        [
            {
                "title": f"piece-{index}",
                "xml_score": str(score),
                "audio_performance": str(tmp_path / f"{index}.wav"),
                "match": str(tmp_path / f"{index}.match"),
            }
            for index in range(1, 4)
        ]
    )
    monkeypatch.setattr(runner, "load_metadata", lambda dataset: metadata)
    monkeypatch.setattr(runner, "print_summary_table", lambda results: results)
    failed = set()

    def follow(score_path, audio_path, config, **kwargs):
        index = int(audio_path.stem)
        if index in failed:
            raise RuntimeError("piece failed")
        if not kwargs["dry_run"]:
            # Exercise the actual pooled summary with a small exact alignment.
            values = np.column_stack([np.arange(11.0), np.arange(11.0)])
            for prefix in ("wp", "gt"):
                path = kwargs["save_dir"] / f"{prefix}_{kwargs['run_name']}.tsv"
                np.savetxt(
                    path,
                    values,
                    delimiter="\t",
                    header="perf_sec\tscore_beat",
                    comments="",
                )
        return {"tracked": True}

    monkeypatch.setattr(runner, "run_score_following", follow)
    return failed


@pytest.mark.parametrize("indices,expected", [(None, [1, 2, 3]), ([3, 1], [1, 3])])
def test_serial_and_parallel_keep_metadata_order(
    pieces, monkeypatch, indices, expected
):
    # Threads let the stub run concurrently without loading real audio in children.
    worker_counts = []

    def pool(*, max_workers):
        worker_counts.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(runner, "ProcessPoolExecutor", pool)
    config = runner.AudioEvalConfig(method="arzt", dataset="asap")
    serial, parallel = [
        runner.run_tests_and_eval_by_dataset(
            "asap", config, dry_run=True, indices=indices, workers=workers
        )
        for workers in (1, 2)
    ]
    assert serial == parallel
    assert serial["Index"] == expected
    assert worker_counts == [2]


@pytest.mark.parametrize(
    "workers,indices,message",
    [
        (0, None, "workers must be positive"),
        (-1, None, "workers must be positive"),
        (1, [0], "one-based"),
        (2, [4], "one-based"),
        (2, [1, 1], "distinct"),
    ],
)
def test_invalid_selection_is_rejected(pieces, workers, indices, message):
    config = runner.AudioEvalConfig(method="arzt", dataset="asap")
    with pytest.raises(ValueError, match=message):
        runner.run_tests_and_eval_by_dataset(
            "asap", config, dry_run=True, indices=indices, workers=workers
        )


@pytest.mark.parametrize("failed", [{2}, {1, 2, 3}])
@pytest.mark.parametrize("sweep", [False, True])
def test_failed_run_saves_and_reports_summaries_before_raising(
    pieces, tmp_path, monkeypatch, failed, sweep
):
    pieces.update(failed)
    output = tmp_path / "output"
    args = runner.build_parser().parse_args(
        [
            "--dataset",
            "asap",
            "--method",
            "arzt",
            "--no-plots",
            "--output-dir",
            str(output),
        ]
    )
    args.sweep = sweep
    args.wandb = not sweep
    monkeypatch.setattr(
        runner, "wandb", SimpleNamespace(config={}, run=SimpleNamespace(id="test"))
    )
    logged = []

    def check_saved(*summaries):
        assert (output / "summary_all.json").exists()
        assert (output / "summary_tracked.json").exists()
        logged.append(summaries)

    monkeypatch.setattr(runner, "log_summary", check_saved)
    monkeypatch.setattr(runner, "report_results_to_wandb", check_saved)
    completed = 3 - len(failed)
    with pytest.raises(RuntimeError, match=f"Incomplete evaluation: {completed}/3"):
        runner.main(args)
    assert len(logged) == 1
    assert json.loads((output / "completion.json").read_text()) == {
        "expected": 3,
        "completed": completed,
        "complete": False,
    }
    assert (output / "test_results.tsv").exists()
    for label in ("all", "tracked"):
        summary = json.loads((output / f"summary_{label}.json").read_text())
        assert summary["tracked_count"] == completed


def test_complete_selection_uses_selected_count(pieces, tmp_path):
    output = tmp_path / "output"
    args = runner.build_parser().parse_args(
        [
            "--dataset",
            "asap",
            "--method",
            "arzt",
            "--no-plots",
            "--indices",
            "3",
            "1",
            "--output-dir",
            str(output),
        ]
    )
    runner.main(args)
    assert json.loads((output / "completion.json").read_text()) == {
        "expected": 2,
        "completed": 2,
        "complete": True,
    }
