"""The robot visitor (P6): does the live demo still answer? Run every 6 hours by robot.yml.

A headless browser opens the live page with ?synthetic=1, so the page tags its events and
the dashboard never counts them as real visits. It waits up to 90 s for the model to load,
pastes a passage written for the purpose, and expects an answer within 10 s. A failure
(the site down, the model not loading, no answer, a JavaScript error) gets one retry, then
exits 1: the workflow fails, and GitHub emails the repo's owner. When monitoring is on, the
robot's synthetic events also draw the dashboard's robot line, and the alert fires if it
falls silent for 12 hours.

Usage:  python scripts/robot_visit.py --url https://ryqn-name-that-llm.static.hf.space
        (add --channel msedge to drive the installed Edge instead of Playwright's Chromium)
"""

import argparse
import sys
import time

# Written for the robot, in a chatbot's style: never a competition passage, and not the
# deploy's smoke-test text either, so the two never get confused on the dashboard.
ROBOT_TEXT = (
    "A good night's sleep starts long before bedtime. Morning daylight sets the body's "
    "clock, so a short walk after waking helps more than most people expect. Caffeine "
    "lingers for hours, which is why an afternoon coffee can still matter at midnight. "
    "Keep the bedroom cool, dark and quiet, and save it for sleep rather than work. If you "
    "can't fall asleep within twenty minutes, get up, read something calm in dim light, "
    "and return when you feel drowsy. Above all, keep regular hours, even at weekends: "
    "consistency teaches the body when to wind down."
)


def visit(browser, url: str) -> tuple[float, float, str]:
    """One visit: (seconds to load the model, seconds to answer, the best guess shown)."""
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        start = time.monotonic()
        page.goto(url, timeout=90_000)
        page.wait_for_selector("#identify:not([disabled])", timeout=90_000)
        load_s = time.monotonic() - start
        page.fill("#text", ROBOT_TEXT)
        start = time.monotonic()
        page.click("#identify")
        page.wait_for_selector("#result:not([hidden])", timeout=10_000)
        answer_s = time.monotonic() - start
        best = page.inner_text("#best").replace("\n", " ")
        page.wait_for_timeout(2_000)  # the page's events are fire-and-forget: let them go out
    finally:
        page.close()
    if errors:
        raise RuntimeError(f"JavaScript errors: {errors}")
    return load_s, answer_s, best


def main(argv=None) -> int:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="the live demo's address")
    parser.add_argument("--channel", help="a browser channel, e.g. msedge (default: Chromium)")
    args = parser.parse_args(argv)
    url = args.url.rstrip("/") + "/?synthetic=1"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        for attempt in (1, 2):
            try:
                load_s, answer_s, best = visit(browser, url)
            except Exception as error:  # any failure: a timeout, an error page, a JS error
                print(f"attempt {attempt} failed: {type(error).__name__}: {str(error)[:300]}")
                continue
            print(f"ok: the model loaded in {load_s:.1f} s, the answer took {answer_s:.2f} s")
            print(f"the page answered: {best}")
            browser.close()
            return 0
        browser.close()
    print("FAIL: the live demo didn't answer, twice")
    return 1


if __name__ == "__main__":
    sys.exit(main())
