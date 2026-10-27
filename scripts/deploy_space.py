"""Deploy the free demo to its Hugging Face Space (P4 revision): run by deploy-app.yml.

The public demo is a STATIC Space (design §12): one page that runs `lr` inside the visitor's
browser. On every merge to main that touches the demo, GitHub Actions runs this script:
  1. it downloads the model files of the MLflow run that app/demo/MODEL_RUN pins (DagsHub);
  2. stage_space copies exactly what the Space serves into a fresh folder: the page, the
     engine, the model card, the model files, and version.json (this commit and that run);
  3. the Space is created if needed (static, public), and the folder is uploaded;
  4. wait_until_live polls the site until its version.json names this commit;
  5. smoke_test checks that the site serves exactly the uploaded files, byte for byte (a
     large file served as a Git LFS pointer, or a stale cached copy, would fail here), then
     runs the engine on them under Node, on a passage written for the purpose.

Settings come from the environment (GitHub Secrets and Variables): HF_TOKEN, HF_SPACE
("owner/name"), GITHUB_SHA, and the MLflow server's MLFLOW_TRACKING_URI,
MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD (a DagsHub token, to read the run).
The tokens never reach the logs or the Space.
"""

import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import httpx

from name_that_llm.browser import run_engine

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "HF_TOKEN",
    "HF_SPACE",
    "GITHUB_SHA",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)
PAGE_FILES = ("index.html", "ntl.js", "README.md")  # from app/demo
MODEL_FILES = ("model.json", "weights.bin")  # from the pinned MLflow run

# Written for this smoke test, in the plain style of a chatbot answer. Never a competition
# passage: the rules forbid publishing those.
SMOKE_TEXT = (
    "Learning a new language as an adult is less about talent than about routine. Short "
    "daily sessions work better than long weekly ones, because memory rewards repetition "
    "spaced over time. Start with the words you actually need, practise them out loud, and "
    "accept mistakes as part of the process. Within a few months, simple conversations "
    "become possible, and every exchange builds a little more confidence."
)


def pinned_run(repo_root: Path) -> str:
    """The MLflow run whose model files the demo serves (app/demo/MODEL_RUN)."""
    return (Path(repo_root) / "app" / "demo" / "MODEL_RUN").read_text(encoding="utf-8").strip()


def stage_space(repo_root: Path, model_dir: Path, out_dir: Path, commit: str) -> list[Path]:
    """Copy what the Space serves into `out_dir`; return the staged files (relative paths).

    Exactly: the page, the engine and the model card from app/demo, the two model files,
    and version.json, which the page's footer shows and wait_until_live watches for.
    Nothing else from the repo (above all, never data/) and nothing else from the run.
    """
    demo, out_dir = Path(repo_root) / "app" / "demo", Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in PAGE_FILES:
        shutil.copy2(demo / name, out_dir / name)
    for name in MODEL_FILES:
        shutil.copy2(Path(model_dir) / name, out_dir / name)
    version = {"commit": commit, "model_run": pinned_run(repo_root)}
    (out_dir / "version.json").write_text(json.dumps(version), encoding="utf-8")
    return sorted(p.relative_to(out_dir) for p in out_dir.iterdir())


def _get(url: str) -> httpx.Response:
    """GET, past any cache: the query string is new on every call."""
    return httpx.get(url, params={"t": time.time()}, follow_redirects=True, timeout=60)


def wait_until_live(api, space: str, commit: str, timeout: float = 900, every: float = 15) -> str:
    """Poll until the site serves this commit's version.json; return the site's address."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Ask Hugging Face where the site is: static Spaces live on *.static.hf.space.
        host = api.space_info(space).host
        try:
            live = _get(f"{host}/version.json").json().get("commit") if host else None
        except (httpx.HTTPError, ValueError):  # not served yet, or not JSON yet
            live = None
        if live == commit:
            print(f"live: {host}")
            return host
        print(f"waiting: {host} serves commit {str(live)[:7]}")
        time.sleep(every)
    sys.exit(f"STOP: {commit[:7]} wasn't live after {timeout / 60:.0f} minutes")


def smoke_test(host: str, staged: Path) -> None:
    """The site serves exactly the uploaded files, and the engine works on them."""
    for path in sorted(staged.iterdir()):
        live = _get(f"{host}/{path.name}")
        if live.status_code != 200 or live.content != path.read_bytes():
            sys.exit(
                f"STOP: the site's {path.name} isn't the uploaded file (HTTP "
                f"{live.status_code}, {len(live.content):,} bytes, not {path.stat().st_size:,})"
            )
    # The site's bytes are the staged ones, and the staged ntl.js is the repo's, so the
    # repo's engine on the staged model files is exactly what visitors run.
    answer = run_engine("predict", {"model": str(staged), "texts": [SMOKE_TEXT]})
    probabilities = answer["probabilities"][0]
    # isclose, not abs(...) > tol: a NaN fails isclose, but passes the other comparison.
    if len(probabilities) != 18 or not math.isclose(sum(probabilities), 1, abs_tol=1e-6):
        sys.exit("STOP: the engine didn't give 18 probabilities summing to 1")
    labels = json.loads((staged / "model.json").read_text(encoding="utf-8"))["labels"]
    best = max(range(18), key=probabilities.__getitem__)
    print(f"smoke test: {labels[best]} ({probabilities[best]:.0%}); every file identical")


def main() -> None:
    env = os.environ
    missing = [key for key in REQUIRED if not env.get(key)]
    if missing:
        sys.exit(f"missing settings: {', '.join(missing)} (GitHub Secrets and Variables)")
    import mlflow
    from huggingface_hub import HfApi

    api = HfApi(token=env["HF_TOKEN"])
    space, commit, run = env["HF_SPACE"], env["GITHUB_SHA"], pinned_run(REPO_ROOT)
    with tempfile.TemporaryDirectory() as tmp:
        print(f"downloading the model files of run {run}")
        model_dir = mlflow.artifacts.download_artifacts(run_id=run, dst_path=f"{tmp}/model")
        staged_dir = Path(tmp) / "space"
        staged = stage_space(REPO_ROOT, Path(model_dir), staged_dir, commit)
        api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True, private=False)
        print(f"uploading {len(staged)} files to the Space {space}")
        api.upload_folder(
            repo_id=space,
            folder_path=staged_dir,
            repo_type="space",
            commit_message=f"Deploy {commit[:7]}",
            # Delete what earlier deploys uploaded and this one doesn't (Hugging Face
            # keeps .gitattributes, which says how large files are stored).
            delete_patterns=["*"],
        )
        host = wait_until_live(api, space, commit)
        smoke_test(host, staged_dir)


if __name__ == "__main__":
    main()
