"""The one command a submitter runs.

run_submission.py is the engine and carries twelve CI-oriented flags; a
submitter needs one of them. This front door is locked to the validation fold on
purpose — the eval fold is what the leaderboard measures, and not providing a
convenient way to develop against it is the cheapest way to keep the
declaration in a submission's metadata.yaml true.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_submission as entry

SCRIPT = REPO_ROOT / "matchmaker_eval" / "test_submission.py"


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


class TestItIsLockedToTuning:
    def test_there_is_no_fold_option(self):
        assert entry.VALID_FOLD == "valid"
        assert "--fold" not in run("--help").stdout, (
            "a submitter must not be able to point this at the eval fold"
        )


class TestArguments:
    def test_a_directory_or_a_method_but_not_both(self):
        done = run("submissions/baseline-constant-tempo", "--method", "pthmm")
        assert done.returncode != 0
        assert "not both" in done.stderr

    def test_neither_is_refused(self):
        done = run()
        assert done.returncode != 0
        assert "not both" in done.stderr


class TestTheSummary:
    def metrics(self, **over):
        record = {
            "submission": "demo",
            "input_type": "midi",
            "n_tracked": 3,
            "n_pieces": 4,
            "summary_all": {
                "tracking_rate": 0.75,
                "rtf": 0.21,
                "median_b": 0.4,
            },
            "summary_tracked": {"median_b": 0.05},
            "pieces": [
                {"piece_id": "asap/a/a", "dataset": "asap", "tracked": True},
                {"piece_id": "batik/b/b", "dataset": "batik", "tracked": True},
                {"piece_id": "batik/c/c", "dataset": "batik", "tracked": True},
                {"piece_id": "vienna/d/d", "dataset": "vienna", "tracked": False},
            ],
            "failures": [],
        }
        record.update(over)
        return record

    def test_summary_reports_results_and_failed_pieces(self):
        text = entry.summarise(
            self.metrics(failures=[{"piece_id": "x/y/z", "error": "ValueError: boom"}])
        )
        assert "3/4 pieces" in text and "75%" in text
        assert "0.05" in text and "0.4" in text
        assert "vienna/d/d" in text and "x/y/z" in text

    def test_it_survives_a_run_with_no_tracked_pieces(self):
        record = self.metrics(
            n_tracked=0,
            summary_tracked={},
            pieces=[dict(p, tracked=False) for p in self.metrics()["pieces"]],
        )
        text = entry.summarise(record)
        assert "0/4 pieces" in text
