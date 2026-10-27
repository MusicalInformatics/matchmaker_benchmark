"""Piece selection, parallel dispatch and persistence after a failed piece."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from matchmaker_eval import test_audio as runner


@pytest.fixture
def failed_piece_indices(tmp_path, monkeypatch):
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


def test_serial_and_parallel_keep_metadata_order(failed_piece_indices, monkeypatch):
    # Threads let the stub run concurrently without loading real audio in children.
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPoolExecutor)
    config = runner.AudioEvalConfig(method="arzt", dataset="asap")
    serial, parallel = [
        runner.run_tests_and_eval_by_dataset(
            "asap", config, dry_run=True, indices=[3, 1], workers=workers
        )
        for workers in (1, 2)
    ]
    assert serial == parallel
    assert serial["Index"] == [1, 3]


@pytest.mark.parametrize("failed", [{2}, {1, 2, 3}])
def test_failed_run_saves_and_reports_summaries_before_raising(
    failed_piece_indices, tmp_path, monkeypatch, failed
):
    failed_piece_indices.update(failed)
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
    args.sweep = True
    monkeypatch.setattr(
        runner, "wandb", SimpleNamespace(config={}, run=SimpleNamespace(id="test"))
    )
    logged = []

    def check_saved(*summaries):
        assert (output / "summary_all.json").exists()
        assert (output / "summary_tracked.json").exists()
        logged.append(summaries)

    monkeypatch.setattr(runner, "log_summary", check_saved)
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
