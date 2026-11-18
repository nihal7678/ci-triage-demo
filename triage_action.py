"""
triage_action.py -- runs INSIDE GitHub Actions right after a workflow run fails.

For the failed run it:
  1. reads the failed jobs, and the project's runs of the last 7 days (GitHub API)
  2. builds the model inputs for each failed job   (triage_features.py)
  3. asks the triage API for P(flaky) and a decision (api.py)
  4. writes a table to the run's summary page, comments on the pull request if there is one,
     and -- only if AUTO_RERUN=true -- re-runs the failed jobs.

Safety rules (each one is a design decision, explained in the README):
  * default is DRY RUN: it only reports, it never re-runs anything
  * it never re-runs a run that is already a re-run (no loops)
  * the whole run is re-run only if EVERY failed job says "rerun"
  * if the API is unreachable or answers badly, the answer is "human" (do nothing), and the
    step still finishes -- a broken triage must never make CI worse
Environment: GITHUB_TOKEN, GITHUB_REPOSITORY, RUN_ID, API_URL (default http://127.0.0.1:8000),
TRIAGE_X (default 15), AUTO_RERUN (default false)
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from triage_features import FAILED, count_features, parse_ts

GITHUB_API = "https://api.github.com"
MAX_JOBS = 10          # more failed jobs than this -> leave it to a human
HISTORY_DAYS = 7
MAX_PAGES = 10         # 1000 runs; if a project has more, the history is truncated (we say so)


class GH:
    def __init__(self, token, repo):
        self.repo = repo
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})

    def get(self, path, **params):
        r = self.s.get(f"{GITHUB_API}/repos/{self.repo}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def post(self, path, body=None):
        r = self.s.post(f"{GITHUB_API}/repos/{self.repo}{path}", json=body, timeout=30)
        r.raise_for_status()
        return r


def recent_runs(gh, now):
    since = (now - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs, truncated = [], False
    for page in range(1, MAX_PAGES + 1):
        d = gh.get("/actions/runs", created=f">={since}", status="completed", per_page=100, page=page)
        batch = d.get("workflow_runs", [])
        runs += batch
        if len(batch) < 100:
            break
        truncated = page == MAX_PAGES
    return runs, truncated


def failed_jobs(gh, run_id, attempt):
    d = gh.get(f"/actions/runs/{run_id}/attempts/{attempt}/jobs", per_page=100)
    return [j for j in d.get("jobs", []) if j.get("conclusion") in FAILED]


def job_seconds(job):
    try:
        return max((parse_ts(job["completed_at"]) - parse_ts(job["started_at"])).total_seconds(), 0.0)
    except Exception:
        return 0.0


def predict(api_url, payload, session=requests, retries=2, timeout=5):
    """Returns (answer or None, round-trip ms, error text or None). Never raises."""
    err = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            r = session.post(f"{api_url}/predict", json=payload, timeout=timeout)
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                return r.json(), ms, None
            err = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code < 500:        # our request is wrong; retrying will not help
                return None, ms, err
        except Exception as e:             # connection refused, timeout, ...
            ms = (time.perf_counter() - t0) * 1000
            err = f"{type(e).__name__}: {e}"
        time.sleep(0.2 * (attempt + 1))
    return None, ms, err


def triage(gh, api_url, run_id, X=15.0, auto_rerun=False, session=requests, now=None):
    t_start = time.perf_counter()
    now = now or datetime.now(timezone.utc)
    m = {"run_id": run_id, "status": "ok", "decision": None, "jobs": [], "actions": [],
         "history_truncated": False}

    def finish(status):
        m["status"] = status
        m["total_seconds"] = round(time.perf_counter() - t_start, 3)
        return m

    run = gh.get(f"/actions/runs/{run_id}")
    attempt = run["run_attempt"]
    m["workflow"] = run["name"]
    if run.get("conclusion") not in FAILED:
        return finish("skipped_not_failed")
    if attempt > 1:
        return finish("skipped_already_rerun")
    jobs = failed_jobs(gh, run_id, attempt)
    if not jobs:
        return finish("skipped_no_failed_jobs")

    runs, m["history_truncated"] = recent_runs(gh, now)
    fallback = False
    for job in jobs[:MAX_JOBS]:
        feats = count_features(runs, run["name"], parse_ts(job["completed_at"]), len(jobs))
        payload = {**feats, "job_duration_s": job_seconds(job), "X": X}
        ans, ms, err = predict(api_url, payload, session=session)
        row = {"job": job["name"], "job_seconds": round(payload["job_duration_s"], 1),
               "api_ms": round(ms, 1)}
        if ans is None:
            fallback = True
            row.update(decision="human", p_flaky=None, threshold_p=None, note=f"API failed: {err}")
        else:
            row.update(decision=ans["decision"], p_flaky=ans["p_flaky"], threshold_p=ans["threshold_p"])
        m["jobs"].append(row)

    if len(jobs) > MAX_JOBS:
        m["decision"] = "human"
        m["note"] = f"{len(jobs)} failed jobs (more than {MAX_JOBS}): probably a real problem"
    else:
        m["decision"] = "rerun" if all(j["decision"] == "rerun" for j in m["jobs"]) else "human"

    if m["decision"] == "rerun" and auto_rerun:
        gh.post(f"/actions/runs/{run_id}/rerun-failed-jobs")
        m["actions"].append("re-ran the failed jobs")
    elif m["decision"] == "rerun":
        m["actions"].append("dry run: would have re-run the failed jobs")

    body = summary_markdown(m, X)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(body)
    for pr in run.get("pull_requests", [])[:1]:
        try:
            gh.post(f"/issues/{pr['number']}/comments", {"body": body})
            m["actions"].append(f"commented on PR #{pr['number']}")
        except Exception as e:             # a failed comment must not fail the step
            m["actions"].append(f"could not comment: {type(e).__name__}")
    return finish("fallback" if fallback else "ok")


def summary_markdown(m, X):
    lines = [f"### CI failure triage: **{m['decision'].upper()}**",
             f"Workflow `{m['workflow']}`, run {m['run_id']}. Cost setting X = {X} machine-minutes.", "",
             "| failed job | P(flaky) | needs P above | decision | job length (s) |", "|---|---|---|---|---|"]
    for j in m["jobs"]:
        p = "n/a" if j["p_flaky"] is None else f"{j['p_flaky']:.2f}"
        t = "n/a" if j["threshold_p"] is None else f"{j['threshold_p']:.2f}"
        lines.append(f"| {j['job']} | {p} | {t} | {j['decision']} | {j['job_seconds']} |")
    if m.get("note"):
        lines.append(f"\n{m['note']}")
    if m.get("history_truncated"):
        lines.append("\nWarning: the project has more than 1000 runs in 7 days; history was truncated.")
    lines.append("\n_Prototype: a probability from a small model, not a verdict. "
                 "Re-running is only automatic when AUTO_RERUN is on._\n")
    return "\n".join(lines)


def main():
    repo, run_id = os.environ["GITHUB_REPOSITORY"], os.environ["RUN_ID"]
    gh = GH(os.environ["GITHUB_TOKEN"], repo)
    X = float(os.environ.get("TRIAGE_X") or 15)
    auto = (os.environ.get("AUTO_RERUN") or "false").lower() == "true"
    api_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
    try:
        m = triage(gh, api_url, run_id, X=X, auto_rerun=auto)
    except Exception as e:
        print("TRIAGE_METRICS " + json.dumps({"run_id": run_id, "status": "error", "error": repr(e)}))
        print(f"::error::triage failed: {e!r}")
        sys.exit(1)
    print("TRIAGE_METRICS " + json.dumps(m))
    if m["status"] == "fallback":
        print("::warning::the triage API failed for at least one job; answered 'human' for those")


if __name__ == "__main__":
    main()
