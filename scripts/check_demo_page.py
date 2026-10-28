"""Check the instrumented demo page in headless Edge, before its deploy (P6 Task 4).

Run by hand. Playwright drives Microsoft Edge (headless) on the STAGED page, answering every
request from the staged folder: Avast on this PC cuts large localhost transfers, so the page
is never served over HTTP. A fake collector on the page's own origin records what the page
would send to Grafana. Four cases:
  1. a recording collector, with ?synthetic=1: a paste sends page_load and answer, then the
     challenger's page_load; one feedback sends one `feedback`, and a second click sends
     nothing; no recorded body holds a word of the passage; every event says synthetic;
  2. a collector answering 500, then one whose requests are aborted: the page gives the
     same answer and thanks the visitor, exactly as in case 1 (monitoring can't break it);
  3. a pinned challenger whose files are missing: answers still come, without a
     challenger label, and the failure is reported as an event;
  4. no collector configured (the live page until P6 Task 5): no notice, no "Do you know
     who wrote it?" question, and nothing sent.
None may raise a JavaScript error. The browser's own "Failed to load resource" lines don't
count: cases 2 and 3 provoke them on purpose. Screenshots at 900 and 390 px go to --shots.

Usage:  uv run --all-extras --with playwright python -X utf8 scripts/check_demo_page.py
"""

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORIGIN = "https://demo.test"  # a made-up origin: every request is answered from the folder
COLLECT = f"{ORIGIN}/collect"
# Written for this check, in a chatbot's style. Never a competition passage.
PASSAGE = (
    "Lighthouses guided sailors through treacherous waters long before satellites existed. "
    "Each keeper trimmed the wicks nightly, polished enormous lenses, and logged every "
    "passing vessel in weathered notebooks. When storms rolled in, keepers stayed awake for "
    "days, sounding foghorns and watching for distress flares. Automation arrived slowly: "
    "electric lamps replaced oil, then timers replaced keepers, and finally remote "
    "monitoring made most stations unmanned. Today many towers survive as museums, where "
    "visitors climb spiral staircases and imagine the solitary routines of the people who "
    "once tended the light through the longest winter nights."
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("data/kaggle-outputs/browser-demo"))
    parser.add_argument("--shots", type=Path, default=Path("data/kaggle-outputs/page-check"))
    return parser.parse_args(argv)


def stage(
    out: Path, model: Path, challenger_run: str, with_challenger_files: bool, url: str = COLLECT
) -> Path:
    """The Space's files, staged by the deploy script, with a collector on our origin."""
    spec = importlib.util.spec_from_file_location(
        "deploy_space", REPO / "scripts" / "deploy_space.py"
    )
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    challenger_dir = model if with_challenger_files else None  # the same model, twice
    deploy.stage_space(REPO, model, out, "local", challenger_dir, collector_url=url)
    version = json.loads((out / "version.json").read_text(encoding="utf-8"))
    version["challenger_run"] = challenger_run  # CHALLENGER_RUN is empty in the repo
    (out / "version.json").write_text(json.dumps(version), encoding="utf-8")
    return out


def serve(route, staged: Path):
    """Answer a request for one of the page's files from the staged folder."""
    path = route.request.url.split("?")[0].removeprefix(ORIGIN + "/") or "index.html"
    file = staged / path
    if file.is_file():
        return route.fulfill(path=str(file))
    return route.fulfill(status=404, body="not found")


def run_without_collector(browser, staged: Path) -> dict:
    """A visit to a page deployed without a collector: what's hidden, and what's POSTed."""
    page = browser.new_page()
    errors, posts = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: request.method == "POST" and posts.append(request.url))
    page.route(f"{ORIGIN}/**", lambda route: serve(route, staged))
    page.goto(f"{ORIGIN}/index.html")
    page.wait_for_selector("#identify:not([disabled])", timeout=60_000)
    page.fill("#text", PASSAGE)
    page.click("#identify")
    page.wait_for_selector("#result:not([hidden])")
    page.wait_for_timeout(1000)
    hidden = page.is_hidden("#feedback") and page.is_hidden("#notice")
    page.close()
    return {"errors": errors, "posts": posts, "hidden": hidden}


def run(browser, staged: Path, collector: str, query: str = "", shots: Path | None = None):
    """One visit: paste, identify, answer the question. Returns what it observed."""
    page = browser.new_page(viewport={"width": 900, "height": 1100})
    errors, bodies = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "console",
        lambda msg: (
            msg.type == "error"
            and not msg.text.startswith("Failed to load resource")
            and errors.append(msg.text)
        ),
    )

    def answer(route):
        request = route.request
        if request.url.split("?")[0] == COLLECT:
            if collector == "abort":
                return route.abort()
            bodies.append(request.post_data or "")
            return route.fulfill(status=204 if collector == "record" else 500, body="")
        return serve(route, staged)

    page.route(f"{ORIGIN}/**", answer)
    page.goto(f"{ORIGIN}/index.html{query}")
    page.wait_for_selector("#identify:not([disabled])", timeout=60_000)
    page.fill("#text", PASSAGE)
    page.click("#identify")
    page.wait_for_selector("#result:not([hidden])")
    page.wait_for_timeout(1500)  # the challenger loads after the first answer
    best = page.inner_text("#best")
    page.select_option("#actual", "a person")
    page.click("#send-feedback")
    page.wait_for_selector("#thanks:not([hidden])")
    sent_twice = page.is_enabled("#send-feedback")
    page.wait_for_timeout(500)
    if shots is not None:
        shots.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(shots / "page-900.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 900})
        page.screenshot(path=str(shots / "page-390.png"), full_page=True)
    page.close()
    events = [event for body in bodies if body for event in json.loads(body)["events"]]
    return {"best": best, "errors": errors, "bodies": bodies, "events": events, "again": sent_twice}


def check(condition: bool, what: str, failures: list[str]) -> None:
    print(("ok    " if condition else "FAIL  ") + what)
    if not condition:
        failures.append(what)


def main(argv=None) -> None:
    from playwright.sync_api import sync_playwright

    args = parse_args(argv)
    passage_words = {w.lower() for w in re.findall(r"[^\W\d_]{4,}", PASSAGE)}
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        with_challenger = stage(Path(tmp) / "a", args.model, "challenger-run", True)
        missing_challenger = stage(Path(tmp) / "b", args.model, "missing-run", False)

        print("1. a recording collector, ?synthetic=1")
        one = run(browser, with_challenger, "record", "?synthetic=1", args.shots)
        names = [(e["name"], e["attributes"].get("what", "")) for e in one["events"]]
        check(not one["errors"], f"no JavaScript error {one['errors']}", failures)
        expected = [("page_load", "champion"), ("answer", ""), ("page_load", "challenger")]
        check(names[:3] == expected, f"page_load, answer, challenger page_load: {names}", failures)
        check(names.count(("feedback", "")) == 1, "exactly one feedback", failures)
        check(not one["again"], "the Send button is off once answered", failures)
        answer = next(e for e in one["events"] if e["name"] == "answer")["attributes"]
        check(
            bool(answer.get("challenger_label")) is False,
            "the 1st answer: no challenger yet",
            failures,
        )
        words = {w for body in one["bodies"] for w in re.findall(r"[^\W\d_]+", body.lower())}
        check(
            not passage_words & words,
            f"no word of the passage sent {passage_words & words}",
            failures,
        )
        synthetic = all(e["attributes"].get("synthetic") == "true" for e in one["events"])
        check(synthetic, "every event says synthetic", failures)

        for mode in ("500", "abort"):
            print(f"2. a collector that {'answers 500' if mode == '500' else 'aborts'}")
            two = run(browser, with_challenger, mode)
            check(not two["errors"], f"no JavaScript error {two['errors']}", failures)
            check(two["best"] == one["best"], "the same answer as case 1", failures)

        print("3. a pinned challenger whose files are missing")
        three = run(browser, missing_challenger, "record")
        check(not three["errors"], f"no JavaScript error {three['errors']}", failures)
        check(three["best"] == one["best"], "the same answer as case 1", failures)
        loads = [e["attributes"] for e in three["events"] if e["name"] == "page_load"]
        failed = [a for a in loads if a.get("what") == "challenger" and a.get("ok") == "false"]
        check(len(failed) == 1, "the challenger's failed load is reported once", failures)

        print("4. no collector configured")
        no_collector = stage(Path(tmp) / "c", args.model, "", False, url="")
        four = run_without_collector(browser, no_collector)
        check(not four["errors"], f"no JavaScript error {four['errors']}", failures)
        check(four["hidden"], "no notice, and no question", failures)
        check(not four["posts"], f"nothing sent {four['posts']}", failures)
        browser.close()
    print(f"screenshots: {args.shots}")
    if failures:
        sys.exit(f"{len(failures)} check(s) failed")
    print("every check passed")


if __name__ == "__main__":
    main()
