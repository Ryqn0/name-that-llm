"""Tests for the deploy script (P4 revision): what goes to the public Hugging Face Space.

The upload needs a token and the network, so the live deploy checks it: every live file must
be byte-identical to the uploaded one, and the engine must work on them under Node. What can
break silently is tested here: the wrong files reaching the public Space, above all the
competition data, and the wrong model run named on the page.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_space.py"


@pytest.fixture(scope="module")
def deploy():
    """scripts/ isn't a package, so load the script as a module from its path."""
    pytest.importorskip("huggingface_hub")
    spec = importlib.util.spec_from_file_location("deploy_space", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_stage_space_copies_only_the_demo_files(deploy, tmp_path):
    repo, model = tmp_path / "repo", tmp_path / "model"
    for name in ["index.html", "ntl.js", "cli.mjs", "package.json"]:
        _touch(repo / "app" / "demo" / name)
    _touch(repo / "app" / "demo" / "README.md", "the demo's model card")
    _touch(repo / "app" / "demo" / "MODEL_RUN", "run123\r\n")  # as Python writes it on Windows
    for name in [
        "data/raw/train.csv",
        "app/Dockerfile",
        "app/README.md",
        "README.md",
        "src/name_that_llm/__init__.py",
        "tests/test_x.py",
    ]:
        _touch(repo / name)
    for name in ["model.json", "weights.bin", "anything-else.txt"]:  # the run's download
        _touch(model / name)

    staged = deploy.stage_space(repo, model, tmp_path / "space", commit="abc1234")

    assert sorted(p.as_posix() for p in staged) == [
        "README.md",  # app/demo/README.md: the Space's model card
        "index.html",
        "model.json",
        "ntl.js",
        "version.json",
        "weights.bin",
    ]
    assert (tmp_path / "space" / "README.md").read_text() == "the demo's model card"
    version = json.loads((tmp_path / "space" / "version.json").read_text())
    assert version == {"commit": "abc1234", "model_run": "run123"}
