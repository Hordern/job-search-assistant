"""Minimal, dependency-free recovery server — its only job is restarting
the main dashboard when it's hung. Deliberately NOT built on FastAPI/
uvicorn, which is the stack that hung in the first place (stuck in
graceful shutdown waiting on a background task that never resolved —
confirmed live 2026-10-02). This has no background tasks, no DB access,
nothing async — just a synchronous stdlib HTTP handler — so there's
nothing in here that can hang the same way. Runs as its own always-on
launchd job, independent of the dashboard's lifecycle, on DASHBOARD_PORT
+ 1000 (so e.g. Alice's dashboard on 8000 gets a sidecar on 9000).

Usage: JOBSEARCH_PROFILE=alice python sidecar.py
"""

import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from profile import PROFILE

DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8000"))
SIDECAR_PORT = DASHBOARD_PORT + 1000

PAGE = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Search Assistant recovery — {PROFILE}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 480px; color: #0f172a; }}
button {{ font-size: 1.05rem; padding: 0.75rem 1.25rem; border-radius: 0.5rem; border: none;
  background: #1e293b; color: #fff; cursor: pointer; }}
button:disabled {{ opacity: 0.6; }}
#status {{ margin-top: 1rem; padding: 0.75rem; border-radius: 0.5rem; background: #f1f5f9; }}
</style></head>
<body>
<h1>Dashboard recovery &mdash; {PROFILE}</h1>
<p>Use this only if the main dashboard (port {DASHBOARD_PORT}) is unresponsive. This page is
deliberately separate and minimal so it stays reachable even when the dashboard itself is hung.</p>
<button id="btn" onclick="restart()">Restart dashboard now</button>
<div id="status"></div>
<script>
function restart() {{
  var btn = document.getElementById('btn');
  var status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = 'Restarting…';
  fetch('/restart', {{method: 'POST'}})
    .then(function (r) {{ return r.json(); }})
    .then(function (d) {{ status.textContent = d.message; btn.disabled = false; }})
    .catch(function (e) {{ status.textContent = 'Error: ' + e; btn.disabled = false; }});
}}
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # keep this quiet — it's not the main app's log
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path != "/restart":
            self.send_response(404)
            self.end_headers()
            return
        pid_result = subprocess.run(
            ["lsof", "-iTCP:" + str(DASHBOARD_PORT), "-sTCP:LISTEN", "-t"], capture_output=True, text=True,
        )
        pid = pid_result.stdout.strip().splitlines()[0] if pid_result.stdout.strip() else None
        if pid:
            subprocess.run(["kill", "-9", pid])
            message = f"Killed PID {pid} on port {DASHBOARD_PORT} — launchd should relaunch it within a few seconds."
        else:
            message = f"No process found listening on port {DASHBOARD_PORT} — it may already be down for another reason."
        body = json.dumps({"message": message}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", SIDECAR_PORT), Handler)
    print(f"Recovery sidecar for {PROFILE} listening on 0.0.0.0:{SIDECAR_PORT}")
    server.serve_forever()
