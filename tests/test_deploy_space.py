"""Tests for the deploy script (P4 revision): what goes to the public Hugging Face Space.

The upload needs a token and the network, so the live deploy checks it: the live files must
match the uploaded ones, and the engine must work on them under Node. What can break
silently is tested here: the wrong files reaching the public Space, above all the
competition data; the wrong model run named on the page; and the page check, which must
accept Hugging Face's added tags but nothing else.
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
    for name in ["index.html", "ntl.js", "telemetry.js", "feedback.js", "cli.mjs", "package.json"]:
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
        "config.json",  # the page's settings: the collector's address (P6)
        "feedback.js",
        "index.html",
        "model.json",
        "ntl.js",
        "telemetry.js",
        "version.json",
        "weights.bin",
    ]
    assert (tmp_path / "space" / "README.md").read_text() == "the demo's model card"
    version = json.loads((tmp_path / "space" / "version.json").read_text())
    assert version == {"commit": "abc1234", "model_run": "run123", "challenger_run": ""}


def _demo(repo: Path, challenger_run: str) -> None:
    """The demo's folder: its page files, and the two pins."""
    for name in ["index.html", "ntl.js", "telemetry.js", "feedback.js", "README.md"]:
        _touch(repo / "app" / "demo" / name)
    _touch(repo / "app" / "demo" / "MODEL_RUN", "run123\n")
    _touch(repo / "app" / "demo" / "CHALLENGER_RUN", challenger_run)


def test_stage_space_with_a_challenger_and_a_collector(deploy, tmp_path):
    # Shadow mode (P6): the challenger's model files go to challenger/, and version.json
    # names both runs, so every event can say which models answered.
    repo, model, challenger = tmp_path / "repo", tmp_path / "model", tmp_path / "challenger"
    _demo(repo, "run456\n")
    for folder in (model, challenger):
        for name in ["model.json", "weights.bin"]:
            _touch(folder / name, folder.name)
    url = "https://collector.example/collect/key"
    staged = deploy.stage_space(
        repo, model, tmp_path / "space", "abc1234", challenger_dir=challenger, collector_url=url
    )
    names = {p.as_posix() for p in staged}
    assert {
        "challenger/model.json",
        "challenger/weights.bin",
        "config.json",
        "telemetry.js",
    } <= names
    space = tmp_path / "space"
    assert (space / "challenger" / "model.json").read_text() == "challenger"
    assert (space / "model.json").read_text() == "model"
    version = json.loads((space / "version.json").read_text())
    assert version == {"commit": "abc1234", "model_run": "run123", "challenger_run": "run456"}
    assert json.loads((space / "config.json").read_text()) == {"collector_url": url}


def test_stage_space_without_a_challenger(deploy, tmp_path):
    repo, model = tmp_path / "repo", tmp_path / "model"
    _demo(repo, "")
    for name in ["model.json", "weights.bin"]:
        _touch(model / name)
    deploy.stage_space(repo, model, tmp_path / "space", "abc1234")
    space = tmp_path / "space"
    assert deploy.challenger_run(repo) == "" and not (space / "challenger").exists()
    assert json.loads((space / "version.json").read_text())["challenger_run"] == ""
    assert json.loads((space / "config.json").read_text()) == {"collector_url": ""}


def test_served_intact_allows_only_insertions(deploy):
    # Hugging Face adds tags to a static page's <head> (seen on the live Space: a script with
    # the Space's variables, social-preview <meta> tags). Our page must still be there whole;
    # a stale copy, or one missing a line, must not pass.
    ours = b"<html>\n<head>\n<title>x</title>\n</head>\n<body>new</body>\n</html>\n"
    live = (
        b"<html>\n<head><script>window.huggingface={variables:{}};</script>\n"
        b'<title>x</title>\n<meta property="og:title" content="t"></head>\n'
        b"<body>new</body>\n</html>\n"
    )
    assert deploy.served_intact(ours, live)
    assert not deploy.served_intact(ours, live.replace(b"new", b"old"))
    assert not deploy.served_intact(ours, live.replace(b"<title>x</title>\n", b""))
