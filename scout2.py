"""
scout2.py  --  Step 3b: measure how many re-runs are GENUINE.

Problem found in the pilot: most "re-runs" are not retries of a failure. They are
maintainer approvals (first attempt = action_required, nothing ran). So we check,
for the most promising projects, how the FIRST attempt of each re-run really ended,
and rank projects by their estimated number of genuine re-runs.

What this program does:
  1. Reads data/scout_results.csv (from scout.py) and takes the top candidates
     of each language.
  2. For each project: looks at the same 100-run sample as before, picks up to
     25 re-run runs, and asks GitHub how attempt 1 of each one ended.
  3. Saves data/scout2_results.csv (resumable) and writes a NEW
     data/selected_repos.csv (the old one is kept as selected_repos_v1.csv).

How to run:
    python scout2.py          (about 20-40 minutes; safe to stop and run again)
"""

import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

import collect  # re-uses the token, the careful request function and EXCLUDE

IN_FILE = os.path.join("data", "scout_results.csv")
OUT_FILE = os.path.join("data", "scout2_results.csv")
SELECTED = os.path.join("data", "selected_repos.csv")
BACKUP = os.path.join("data", "selected_repos_v1.csv")

# ----- Settings you can change -------------------------------------------------
SAMPLE_FROM = "2026-09-01"       # same sample window as scout.py
SAMPLE_TO = "2026-09-14"
MIN_TOTAL_RUNS = 300
MIN_RERUNS_IN_SAMPLE = 3
CANDIDATES_PER_LANGUAGE = 25     # how many projects per language to examine
MAX_CHECKS = 25                  # re-run runs to check per project
SELECT_PER_LANGUAGE = 6          # projects to pick per language at the end
MIN_GENUINE_IN_SAMPLE = 2        # at least this many genuine re-runs in the checked ones
# -------------------------------------------------------------------------------

COLUMNS = ["repo", "language", "total_runs_in_window", "rerun_runs_in_sample",
           "checked", "attempt1_failed", "attempt1_action_required",
           "attempt1_other", "errors", "est_reruns", "est_genuine_reruns"]
FAILED_RUN = ("failure", "timed_out")


def choose_candidates(rows):
    by_lang = defaultdict(list)
    for r in rows:
        if (r["repo"] in collect.EXCLUDE
                or int(r["total_runs_in_window"]) < MIN_TOTAL_RUNS
                or int(r["reruns_in_sample"]) < MIN_RERUNS_IN_SAMPLE):
            continue
        by_lang[r["language"]].append(r)
    out = []
    for lang, items in by_lang.items():
        items.sort(key=lambda x: int(x["est_reruns_in_window"]), reverse=True)
        out.extend(items[:CANDIDATES_PER_LANGUAGE])
    return out


def examine(repo, language, total_runs):
    r = collect.api(f"{collect.API}/repos/{repo}/actions/runs",
                    params={"created": f"{SAMPLE_FROM}..{SAMPLE_TO}",
                            "status": "completed", "per_page": 100})
    runs = r.json().get("workflow_runs", []) if r.status_code == 200 else []
    n = len(runs)
    rerun = [x for x in runs
             if x["run_attempt"] > 1 and x["conclusion"] in ("success", "failure")]
    random.shuffle(rerun)
    tally = Counter()
    for x in rerun[:MAX_CHECKS]:
        a1 = collect.api(f"{collect.API}/repos/{repo}/actions/runs/{x['id']}/attempts/1")
        if a1.status_code != 200:
            tally["error"] += 1
            continue
        c = a1.json().get("conclusion")
        if c in FAILED_RUN:
            tally["failed"] += 1
        elif c == "action_required":
            tally["action_required"] += 1
        else:
            tally["other"] += 1
    checked = tally["failed"] + tally["action_required"] + tally["other"]
    est_reruns = round(total_runs * len(rerun) / n) if n else 0
    rate = tally["failed"] / checked if checked else 0
    return {"repo": repo, "language": language, "total_runs_in_window": total_runs,
            "rerun_runs_in_sample": len(rerun), "checked": checked,
            "attempt1_failed": tally["failed"],
            "attempt1_action_required": tally["action_required"],
            "attempt1_other": tally["other"], "errors": tally["error"],
            "est_reruns": est_reruns, "est_genuine_reruns": round(est_reruns * rate)}


def main():
    random.seed(7)
    if not os.path.exists(IN_FILE):
        sys.exit(f"Could not find {IN_FILE}. Run scout.py first.")
    with open(IN_FILE, encoding="utf-8") as f:
        candidates = choose_candidates(list(csv.DictReader(f)))
    print(f"{len(candidates)} candidate projects to examine.")

    done = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding="utf-8") as f:
            done = {r["repo"] for r in csv.DictReader(f)}
        print(f"Found earlier progress: {len(done)} already examined.")
    todo = [c for c in candidates if c["repo"] not in done]

    is_new = not os.path.exists(OUT_FILE)
    with open(OUT_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            w.writeheader()
        for i, c in enumerate(todo, 1):
            try:
                row = examine(c["repo"], c["language"], int(c["total_runs_in_window"]))
            except Exception as e:  # network problem: skip, it is retried next run
                print(f"  Skipping {c['repo']} for now ({type(e).__name__}).")
                continue
            w.writerow(row)
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done (last: {row['repo']}: "
                      f"{row['attempt1_failed']} genuine of {row['checked']} checked)")

    # ---- Summary and new selection ----
    with open(OUT_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tot_checked = sum(int(r["checked"]) for r in rows)
    tot_failed = sum(int(r["attempt1_failed"]) for r in rows)
    tot_appr = sum(int(r["attempt1_action_required"]) for r in rows)
    if tot_checked:
        print(f"\nAcross {len(rows)} projects, {tot_checked} re-run runs were checked:")
        print(f"  first attempt FAILED (genuine)   : {tot_failed} ({100 * tot_failed / tot_checked:.0f}%)")
        print(f"  first attempt action_required    : {tot_appr} ({100 * tot_appr / tot_checked:.0f}%)")
        print(f"  other                            : {tot_checked - tot_failed - tot_appr}")

    good = [r for r in rows if int(r["attempt1_failed"]) >= MIN_GENUINE_IN_SAMPLE]
    by_lang = defaultdict(list)
    for r in good:
        by_lang[r["language"]].append(r)
    chosen = []
    print(f"\n{'language':<12} {'passing':>8} {'chosen':>7}")
    for lang in sorted(by_lang):
        items = sorted(by_lang[lang], key=lambda x: int(x["est_genuine_reruns"]), reverse=True)
        chosen.extend(items[:SELECT_PER_LANGUAGE])
        print(f"{lang:<12} {len(items):>8} {min(len(items), SELECT_PER_LANGUAGE):>7}")
    if not chosen:
        sys.exit("No project passed the rules. Tell me and we will relax them.")

    if os.path.exists(SELECTED) and not os.path.exists(BACKUP):
        shutil.copy(SELECTED, BACKUP)
    chosen.sort(key=lambda x: (x["language"], -int(x["est_genuine_reruns"])))
    with open(SELECTED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(chosen)

    print(f"\nSelected {len(chosen)} projects -> {SELECTED}  (old list kept as {BACKUP})\n")
    for r in chosen:
        print(f"  {r['language']:<11} {r['repo']:<40} est_genuine={r['est_genuine_reruns']:<5} "
              f"genuine {r['attempt1_failed']}/{r['checked']} checked "
              f"(approvals {r['attempt1_action_required']})")


if __name__ == "__main__":
    main()
