"""Deploy the live app to its Hugging Face Space (P4): run by .github/workflows/deploy-app.yml.

On every merge to main that touches the app, GitHub Actions runs this script:
  1. stage_space copies exactly what the Space needs into a fresh folder. It also writes
     the commit into the package (BUILD_COMMIT), so it's baked into the image: /version
     can then never claim a newer commit than the image actually running.
  2. The Space is created if needed (Docker, public), and its settings are written: the
     MLflow registry's address and user as variables, the DagsHub token as a secret.
  3. The staged folder is uploaded. Hugging Face rebuilds the image and restarts the app.
  4. wait_until_live polls until the NEW build answers: /version names this commit and
     /health says it's ready. Building, then loading the models, take several minutes.
  5. smoke_test sends one passage, written for the purpose, and checks the answer's shape.

Settings come from the environment (GitHub Secrets and Variables): HF_TOKEN, HF_SPACE
("owner/name"), GITHUB_SHA, MLFLOW_TRACKING_URI, DAGSHUB_USER, DAGSHUB_TOKEN, and optionally
ENCODER_FOLDS (the design's speed fallback). The tokens go to Hugging Face, never to logs.
"""

import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "HF_TOKEN",
    "HF_SPACE",
    "GITHUB_SHA",
    "MLFLOW_TRACKING_URI",
    "DAGSHUB_USER",
    "DAGSHUB_TOKEN",
)

# Written for this smoke test, in the plain style of a chatbot answer. Never a competition
# passage: the rules forbid publishing those, and this text travels to a public Space.
SMOKE_TEXT = (
    "Learning a new language as an adult is less about talent than about routine. Short "
    "daily sessions work better than long weekly ones, because memory rewards repetition "
    "spaced over time. Start with the words you actually need, practise them out loud, and "
    "accept mistakes as part of the process. Within a few months, simple conversations "
    "become possible, and every exchange builds a little more confidence."
)


def stage_space(repo_root: Path, out_dir: Path, commit: str) -> list[Path]:
    """Copy what the Space needs into `out_dir`; return the staged files (relative paths).

    Exactly: the Dockerfile and the model card (moved from app/ to where Hugging Face
    expects them), the project and lock files, and the package source. Never data/,
    tests or notebooks.
    """
    repo_root, out_dir = Path(repo_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / "app" / "Dockerfile", out_dir / "Dockerfile")
    shutil.copy2(repo_root / "app" / "README.md", out_dir / "README.md")
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(repo_root / name, out_dir / name)
    shutil.copytree(
        repo_root / "src",
        out_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "BUILD_COMMIT"),
        dirs_exist_ok=True,
    )
    (out_dir / "src" / "name_that_llm" / "BUILD_COMMIT").write_text(commit)
    return sorted(p.relative_to(out_dir) for p in out_dir.rglob("*") if p.is_file())


def space_host(space_id: str) -> str:
    """Where Hugging Face serves a Space: "owner/name" -> https://owner-name.hf.space,
    lowercase, with every run of other characters turned into one dash."""
    slug = re.sub(r"[^a-z0-9]+", "-", space_id.lower()).strip("-")
    return f"https://{slug}.hf.space"


def wait_until_live(host: str, commit: str, timeout: float = 1800, every: float = 30) -> dict:
    """Poll until the build of `commit` is up and its model loaded; return its /version."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            version = httpx.get(f"{host}/version", timeout=30).json()
            health = httpx.get(f"{host}/health", timeout=30)
            if version.get("commit") == commit:
                if health.status_code == 200:
                    print(f"live: {version}")
                    return version
                if health.status_code == 500:
                    sys.exit(f"STOP: the new build failed to load the model: {health.json()}")
            running = str(version.get("commit"))[:7]
            print(f"waiting: running commit {running}, health {health.status_code}")
        except (httpx.HTTPError, ValueError) as exc:  # still building: no answer, or not JSON
            print(f"waiting: {type(exc).__name__}")
        time.sleep(every)
    sys.exit(f"STOP: the build of {commit[:7]} wasn't live after {timeout / 60:.0f} minutes")


def smoke_test(host: str) -> None:
    """One real prediction: 18 models, sorted, probabilities summing to 1."""
    response = httpx.post(f"{host}/predict", json={"text": SMOKE_TEXT}, timeout=180)
    if response.status_code != 200:
        sys.exit(f"STOP: /predict answered {response.status_code}: {response.text[:300]}")
    body = response.json()
    probabilities = [entry["probability"] for entry in body["ranking"]]
    if len(probabilities) != 18:
        sys.exit(f"STOP: {len(probabilities)} models in the ranking, not 18")
    if probabilities != sorted(probabilities, reverse=True) or abs(sum(probabilities) - 1) > 1e-3:
        sys.exit("STOP: the ranking isn't sorted, or its probabilities don't sum to 1")
    best = body["ranking"][0]
    print(f"smoke test: {best['model']} ({best['probability']:.0%}) in {body['seconds']} s")


def main() -> None:
    env = os.environ
    missing = [key for key in REQUIRED if not env.get(key)]
    if missing:
        sys.exit(f"missing settings: {', '.join(missing)} (GitHub Secrets and Variables)")
    from huggingface_hub import HfApi

    api = HfApi(token=env["HF_TOKEN"])
    space, commit = env["HF_SPACE"], env["GITHUB_SHA"]
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    api.add_space_variable(space, "MLFLOW_TRACKING_URI", env["MLFLOW_TRACKING_URI"])
    api.add_space_variable(space, "MLFLOW_TRACKING_USERNAME", env["DAGSHUB_USER"])
    api.add_space_secret(space, "MLFLOW_TRACKING_PASSWORD", env["DAGSHUB_TOKEN"])
    if env.get("ENCODER_FOLDS"):
        api.add_space_variable(space, "ENCODER_FOLDS", env["ENCODER_FOLDS"])
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage_space(REPO_ROOT, Path(tmp), commit)
        print(f"uploading {len(staged)} files to the Space {space}")
        api.upload_folder(
            repo_id=space,
            folder_path=tmp,
            repo_type="space",
            commit_message=f"Deploy {commit[:7]}",
            # Remove source files that no longer exist; the rest are overwritten anyway.
            delete_patterns=["src/**"],
        )
    host = space_host(space)
    wait_until_live(host, commit)
    smoke_test(host)


if __name__ == "__main__":
    main()
