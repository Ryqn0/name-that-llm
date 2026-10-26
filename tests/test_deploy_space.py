"""Tests for the deploy script (P4): what goes to the Hugging Face Space, and where it lives.

The upload itself needs a token and the network, so it's checked by the live deploy. What
can break silently is tested here: the wrong files reaching the Space (above all the
competition data), and the wrong address for the live checks.
"""

import importlib.util
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


def test_stage_space_copies_only_the_app_files(deploy, tmp_path):
    repo = tmp_path / "repo"
    wanted = [
        "app/Dockerfile",
        "app/README.md",
        "pyproject.toml",
        "uv.lock",
        "src/name_that_llm/__init__.py",
        "src/name_that_llm/static/index.html",
    ]
    unwanted = [
        "data/raw/train.csv",
        "tests/test_x.py",
        "notebooks/n/n.ipynb",
        "README.md",
        "src/name_that_llm/__pycache__/x.cpython-313.pyc",
        ".git/config",
    ]
    for name in wanted + unwanted:
        _touch(repo / name)
    _touch(repo / "app/README.md", "the model card")  # unlike the project's own README.md

    staged = deploy.stage_space(repo, tmp_path / "space", commit="abc1234")

    assert sorted(p.as_posix() for p in staged) == [
        "Dockerfile",  # app/Dockerfile, moved to where Hugging Face expects it
        "README.md",  # app/README.md: the model card, not the project README
        "pyproject.toml",
        "src/name_that_llm/BUILD_COMMIT",
        "src/name_that_llm/__init__.py",
        "src/name_that_llm/static/index.html",
        "uv.lock",
    ]
    assert (tmp_path / "space" / "src/name_that_llm/BUILD_COMMIT").read_text() == "abc1234"
    assert (tmp_path / "space" / "README.md").read_text() == "the model card"


def test_space_host_follows_hf_rules(deploy):
    # A Space "owner/name" is served at https://owner-name.hf.space: lowercase, and any
    # character other than a letter or a digit becomes a dash.
    assert deploy.space_host("Ryqn0/name-that-llm") == "https://ryqn0-name-that-llm.hf.space"
    assert deploy.space_host("a_b/c.d") == "https://a-b-c-d.hf.space"
