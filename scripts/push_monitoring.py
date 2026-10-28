"""Push the demo's monitoring to Grafana Cloud: the dashboard and the alert rules (P6).

Monitoring as code: monitoring/dashboard.json and monitoring/alerts.json are the source, and
this script sends them to the stack, replacing what's there. Run by hand, after a change:
  1. creates the folder "Name That LLM" (or keeps the one that exists);
  2. sends the dashboard (/api/dashboards/db), overwriting the previous version;
  3. sends each alert rule (/api/v1/provisioning/alert-rules), updating it if it exists.
The alert rules name their Loki data source by uid, which differs per stack: the script
looks it up, and fills in the rules' ${LOKI_UID}. The dashboard asks for its data sources
through variables, so it needs no uid.

Settings come from the environment: GRAFANA_URL (the stack, e.g. https://<name>.grafana.net)
and GRAFANA_TOKEN (a service-account token, role Editor). The token goes in a header only:
never in a body, never printed.

Usage:  python scripts/push_monitoring.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "name-that-llm"


def build_requests(
    dashboard: dict, alerts: list[dict], folder_uid: str
) -> list[tuple[str, str, dict]]:
    """The API calls that install the dashboard and the alerts: (method, path, JSON body)."""
    requests = [("POST", "/api/folders", {"uid": folder_uid, "title": "Name That LLM"})]
    body = {"dashboard": dashboard, "folderUid": folder_uid, "overwrite": True}
    requests.append(("POST", "/api/dashboards/db", body | {"message": "push_monitoring.py"}))
    for rule in alerts:
        rule_body = rule | {"folderUID": folder_uid}
        requests.append(("POST", "/api/v1/provisioning/alert-rules", rule_body))
    return requests


def main() -> None:
    import httpx

    url, token = os.environ.get("GRAFANA_URL"), os.environ.get("GRAFANA_TOKEN")
    if not url or not token:
        sys.exit("missing GRAFANA_URL or GRAFANA_TOKEN (see this script's docstring)")
    dashboard = json.loads((ROOT / "monitoring" / "dashboard.json").read_text(encoding="utf-8"))
    alerts_text = (ROOT / "monitoring" / "alerts.json").read_text(encoding="utf-8")
    # X-Disable-Provenance: the rules stay editable in Grafana's interface.
    headers = {"Authorization": f"Bearer {token}", "X-Disable-Provenance": "true"}
    with httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=30) as client:
        sources = client.get("/api/datasources").raise_for_status().json()
        # Grafana Cloud names its logs data source grafanacloud-<stack>-logs.
        loki = [s for s in sources if s["type"] == "loki"]
        loki.sort(key=lambda s: not s["name"].endswith("-logs"))
        if not loki:
            sys.exit("no Loki data source on this stack: is Frontend Observability set up?")
        print(f"Loki data source: {loki[0]['name']}")
        alerts = json.loads(alerts_text.replace("${LOKI_UID}", loki[0]["uid"]))
        for method, path, body in build_requests(dashboard, alerts, FOLDER):
            response = client.request(method, path, json=body)
            if path == "/api/folders" and response.status_code in (409, 412):
                print("folder: already there")
                continue
            if path.endswith("/alert-rules") and response.status_code in (400, 409):
                # The rule exists: update it in place, under the same uid.
                response = client.put(f"{path}/{body['uid']}", json=body)
            if response.is_error:
                sys.exit(f"STOP: {method} {path} answered HTTP {response.status_code}")
            print(f"{method} {path}: HTTP {response.status_code}")
    print(f"done: {url.rstrip('/')}/d/{dashboard['uid']}")


if __name__ == "__main__":
    main()
