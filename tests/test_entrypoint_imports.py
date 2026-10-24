"""Legacy tools support package imports and direct execution from any directory."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULES = (
    "download_data",
    "estimate_perf_tempo",
    "eval_symbolic",
    "preprocess_urmp",
    "preprocess_winterreise",
    "symbolic_alignment",
    "test_audio",
    "test_symbolic",
)


@pytest.mark.parametrize("mode", ["module", "script"])
def test_tools_import_without_bare_local_modules(tmp_path, mode):
    # Keep expensive third-party imports cached, but isolate each tool's paths.
    code = """
import importlib, runpy, sys
from pathlib import Path
root = Path(sys.argv[1])
mode = sys.argv[2]
base = [p for p in sys.path if Path(p).resolve() not in (root, root / 'matchmaker_eval')]
for name in sys.argv[3:]:
    for key in tuple(sys.modules):
        if key == 'matchmaker_eval' or key.startswith('matchmaker_eval.'):
            del sys.modules[key]
    sys.path[:] = [str(root if mode == 'module' else root / 'matchmaker_eval'), *base]
    if mode == 'module':
        importlib.import_module('matchmaker_eval.' + name)
    else:
        runpy.run_path(str(root / 'matchmaker_eval' / (name + '.py')), run_name='_review')
"""
    done = subprocess.run(
        [sys.executable, "-c", code, str(REPO_ROOT), mode, *MODULES],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("mode", ["module", "script"])
@pytest.mark.parametrize("name", ["download_data", "test_audio"])
def test_help_works_for_both_entrypoints(mode, name):
    target = (
        ["-m", f"matchmaker_eval.{name}"]
        if mode == "module"
        else [str(REPO_ROOT / "matchmaker_eval" / f"{name}.py")]
    )
    done = subprocess.run(
        [sys.executable, *target, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr
    assert "--dataset" in done.stdout
