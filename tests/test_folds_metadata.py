"""Metadata paths and audio-only folds must work without downloading a corpus."""

import csv
import json

import pytest

from matchmaker_eval import folds, leaderboard


def write_metadata(root, rows):
    directory = root / "demo"
    directory.mkdir()
    with (directory / "metadata-demo.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_data_repo_metadata_reads_both_rows_and_resolves_paths(tmp_path, monkeypatch):
    rows = [
        {
            "score": f" demo/score/{name}.musicxml ",
            "audio": f" demo/audio/{name}.mp3 ",
            "midi": f"demo/midi/{name}.mid",
            "annotations": f"demo/annotations/{name}.tsv",
            "estimated_bpm": str(tempo),
        }
        for name, tempo in (("first", 90), ("second", 120))
    ]
    write_metadata(tmp_path, rows)
    monkeypatch.setattr(folds, "DATA_ROOT", tmp_path)
    loaded = folds.data_repo_metadata("demo")
    assert [row["title"] for row in loaded] == ["first", "second"]
    for raw, row in zip(rows, loaded):
        assert row["xml_score"] == str(tmp_path / raw["score"].strip())
        assert row["audio_performance"] == str(tmp_path / raw["audio"].strip())
        assert row["midi_performance"] == str(tmp_path / raw["midi"])
        assert row["performance_annotations"] == str(tmp_path / raw["annotations"])
        assert row["estimated_bpm"] == raw["estimated_bpm"]


@pytest.mark.parametrize("audio", [None, "   "])
def test_metadata_title_falls_back_to_midi(tmp_path, monkeypatch, audio):
    row = {"score": "demo/score/music.musicxml", "midi": " demo/midi/take.mid "}
    if audio is not None:
        row["audio"] = audio
    write_metadata(tmp_path, [row])
    monkeypatch.setattr(folds, "DATA_ROOT", tmp_path)
    result = folds.data_repo_metadata("demo")[0]
    assert result["title"] == "take"
    assert result["midi_performance"] == str(tmp_path / "demo/midi/take.mid")
    assert "audio_performance" not in result


def test_missing_data_repo_metadata_names_the_path(tmp_path, monkeypatch):
    monkeypatch.setattr(folds, "DATA_ROOT", tmp_path)
    with pytest.raises(folds.FoldError, match="metadata-demo.csv"):
        folds.data_repo_metadata("demo")


def test_beyond_piano_is_audio_only_and_separate_from_leaderboard():
    pieces = folds.load_fold("beyond_piano", input_type="audio")
    assert {piece.dataset for piece in pieces} == set(folds.BEYOND_PIANO_DATASETS)
    assert len({piece.piece_id for piece in pieces}) == len(pieces)
    assert all(
        piece.audio_performance and not piece.midi_performance for piece in pieces
    )
    assert all(piece.match.endswith(".tsv") for piece in pieces)
    eval_ids = {piece.piece_id for piece in folds.load_fold("eval")}
    assert eval_ids.isdisjoint(piece.piece_id for piece in pieces)
    with pytest.raises(folds.FoldError, match="no midi performances"):
        folds.load_fold("beyond_piano", input_type="midi")


def test_beyond_piano_results_do_not_enter_leaderboard(tmp_path):
    run = tmp_path / "reference"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"fold": "beyond_piano"}))
    rows, skipped = leaderboard.collect(tmp_path)
    assert rows == []
    assert len(skipped) == 1
    assert "beyond_piano" in skipped[0]
