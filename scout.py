"""
scout.py  --  Step 2: find busy projects that have "run again" history.

What this program does, in plain words:
  1. Asks GitHub for the most popular, recently active projects in several
     programming languages (the "candidates").
  2. For each candidate, looks at ONE sample of 100 checks from 1-14 Sept 2026.
  3. Counts: how many checks the project runs, how many failed,
     how many were run again.
  4. Saves a ranked list to data/scout_results.csv.

It only READS public data. It saves its progress as it goes, so if it stops
(or you close the window) just run it again: it continues where it stopped.

How to run (from the folder that contains token.txt):
    python scout.py
Expect roughly 20-40 minutes. You can leave it running.
"""

import csv
import os
import sys
import time

import requests

# ----- Settings you can change -------------------------------------------------
LANGUAGES = ["Python", "JavaScript", "TypeScript", "Java", "Go",
             "Rust", "C++", "C#", "PHP", "Ruby"]
PAGES_PER_LANGUAGE = 1          # 1 page = 100 projects per language (1000 total)
MIN_STARS = 1000
PUSHED_AFTER = "2026-10-01"     # only projects that are still active
SAMPLE_FROM = "2026-09-01"      # the sample window for counting runs
SAMPLE_TO = "2026-09-14"
API = "https://api.github.com"
OUT_FILE = os.path.join("data", "scout_results.csv")
# -------------------------------------------------------------------------------


def read_token():
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token.strip()
    if os.path.exists("token.txt"):
        with open("token.txt", encoding="utf-8") as f:
            return f.read().strip()
    sys.exit("Could not find token.txt in this folder.")


HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {read_token()}",
    "X-GitHub-Api-Version": "2022-11-28",
}


def get(url, params=None, allow_404=False):
    """Ask GitHub for something. Waits politely if we are going too fast."""
    failed_tries = 0
    while True:
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=60)
        except requests.exceptions.RequestException as e:
            # Internet dropped for a moment. Wait a bit and try the same request again.
            failed_tries += 1
            if failed_tries > 6:
                raise
            wait = 5 * failed_tries
            print(f"  Connection problem ({type(e).__name__}). "
                  f"Retrying in {wait} s (try {failed_tries}/6) ...")
            time.sleep(wait)
            continue
        failed_tries = 0
        if r.status_code == 401:
            sys.exit("GitHub says the token is not valid (401). Check token.txt.")
        if r.status_code in (404, 409, 451) and allow_404:
            return None  # project has no Actions, is disabled, or is blocked
        if r.status_code in (403, 429):
            if r.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - int(time.time()), 1) + 5
            else:
                wait = int(r.headers.get("Retry-After", 60))  # "slow down" message
            print(f"  GitHub asked us to wait. Waiting {wait} s ...")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(10)
            continue
        r.raise_for_status()
        return r


def find_candidates():
    """Search GitHub for popular, active projects in each language."""
    found = {}  # full_name -> language
    for lang in LANGUAGES:
        query = (f"language:{lang} stars:>{MIN_STARS} pushed:>{PUSHED_AFTER} "
                 f"fork:false archived:false")
        for page in range(1, PAGES_PER_LANGUAGE + 1):
            r = get(f"{API}/search/repositories",
                    params={"q": query, "sort": "stars", "order": "desc",
                            "per_page": 100, "page": page})
            items = r.json().get("items", [])
            for it in items:
                found[it["full_name"]] = lang
            print(f"  {lang}: page {page} -> {len(items)} projects")
            time.sleep(3)  # search is limited to ~30 requests per minute
            if len(items) < 100:
                break
    return found


def sample_repo(full_name):
    """Look at up to 100 finished checks of one project in the sample window."""
    r = get(f"{API}/repos/{full_name}/actions/runs",
            params={"created": f"{SAMPLE_FROM}..{SAMPLE_TO}",
                    "status": "completed", "per_page": 100},
            allow_404=True)
    if r is None:
        return None
    data = r.json()
    runs = data.get("workflow_runs", [])
    total = data.get("total_count", 0)
    n = len(runs)
    failures = sum(1 for x in runs if x["conclusion"] == "failure")
    reruns = sum(1 for x in runs if x["run_attempt"] > 1)
    # Rough guess of how many re-run checks the project has in the sample window
    est_reruns = round(total * reruns / n) if n else 0
    return {"total_runs_in_window": total, "sampled": n,
            "failures_in_sample": failures, "reruns_in_sample": reruns,
            "est_reruns_in_window": est_reruns}


def main():
    os.makedirs("data", exist_ok=True)

    # Which projects did we already scout? (so we can continue after a stop)
    done = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding="utf-8") as f:
            done = {row["repo"] for row in csv.DictReader(f)}
        print(f"Found earlier progress: {len(done)} projects already scouted.\n")

    print("Step 1: finding candidate projects ...")
    candidates = find_candidates()
    todo = [(name, lang) for name, lang in candidates.items() if name not in done]
    print(f"\n{len(candidates)} candidates, {len(todo)} still to scout.\n")

    columns = ["repo", "language", "total_runs_in_window", "sampled",
               "failures_in_sample", "reruns_in_sample", "est_reruns_in_window"]
    is_new = not os.path.exists(OUT_FILE)
    with open(OUT_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            w.writeheader()
        print("Step 2: scouting each project (one request each) ...")
        for i, (name, lang) in enumerate(todo, 1):
            try:
                result = sample_repo(name)
            except requests.exceptions.RequestException:
                # Still failing after retries: skip it now, it will be retried next run.
                print(f"  Skipping {name} for now (connection kept failing).")
                continue
            if result is None:
                result = {"total_runs_in_window": 0, "sampled": 0,
                          "failures_in_sample": 0, "reruns_in_sample": 0,
                          "est_reruns_in_window": 0}
            w.writerow({"repo": name, "language": lang, **result})
            f.flush()  # save right away
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done (last: {name}, "
                      f"reruns in sample: {result['reruns_in_sample']})")

    # Ranked copy for easy reading
    with open(OUT_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda x: int(x["est_reruns_in_window"]), reverse=True)
    ranked = os.path.join("data", "scout_ranked.csv")
    with open(ranked, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. Open {ranked} in Excel (the best projects are at the top).")
    print("Top 15 by estimated re-runs:")
    for row in rows[:15]:
        print(f"  {row['repo']:<40} {row['language']:<11} "
              f"runs={row['total_runs_in_window']:<6} "
              f"reruns_in_sample={row['reruns_in_sample']}")


if __name__ == "__main__":
    main()
