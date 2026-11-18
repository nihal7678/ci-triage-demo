"""
select_repos.py  --  Step 3: choose which projects we will collect data from.

What this program does, in plain words:
  1. Reads data/scout_results.csv (made by scout.py).
  2. Keeps only projects with enough activity, failures and re-runs.
  3. Takes the best few projects from EACH language, so we get a mix.
  4. Saves the choice to data/selected_repos.csv and prints it.

You can open data/selected_repos.csv in Excel and delete any row you do not
want (for example, a project that is not real software). The next program
will use whatever rows are left in that file.

How to run:
    python select_repos.py
"""

import csv
import os
import sys
from collections import defaultdict

IN_FILE = os.path.join("data", "scout_results.csv")
OUT_FILE = os.path.join("data", "selected_repos.csv")

# ----- Rules you can change ----------------------------------------------------
MIN_TOTAL_RUNS = 300       # the project runs at least this many checks in the window
MIN_FAILURES = 5           # at least this many failures in the 100-check sample
MIN_RERUNS = 3             # at least this many re-runs in the 100-check sample
PER_LANGUAGE = 6           # how many projects to take from each language
# -------------------------------------------------------------------------------


def main():
    if not os.path.exists(IN_FILE):
        sys.exit(f"Could not find {IN_FILE}. Run scout.py first.")

    with open(IN_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Read {len(rows)} scouted projects.")

    good = [r for r in rows
            if int(r["total_runs_in_window"]) >= MIN_TOTAL_RUNS
            and int(r["failures_in_sample"]) >= MIN_FAILURES
            and int(r["reruns_in_sample"]) >= MIN_RERUNS]
    print(f"{len(good)} projects pass the rules "
          f"(runs>={MIN_TOTAL_RUNS}, failures>={MIN_FAILURES}, reruns>={MIN_RERUNS}).\n")

    by_lang = defaultdict(list)
    for r in good:
        by_lang[r["language"]].append(r)

    chosen = []
    print(f"{'language':<12} {'passing':>8} {'chosen':>7}")
    for lang in sorted(by_lang):
        items = sorted(by_lang[lang],
                       key=lambda x: int(x["est_reruns_in_window"]), reverse=True)
        chosen.extend(items[:PER_LANGUAGE])
        print(f"{lang:<12} {len(items):>8} {min(len(items), PER_LANGUAGE):>7}")

    missing = [l for l in ["Python", "JavaScript", "TypeScript", "Java", "Go",
                           "Rust", "C++", "C#", "PHP", "Ruby"] if l not in by_lang]
    for lang in missing:
        print(f"{lang:<12} {0:>8} {0:>7}   <-- no project passed the rules")

    chosen.sort(key=lambda x: (x["language"], -int(x["est_reruns_in_window"])))
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(chosen[0].keys()))
        w.writeheader()
        w.writerows(chosen)

    print(f"\nSelected {len(chosen)} projects -> {OUT_FILE}\n")
    for r in chosen:
        print(f"  {r['language']:<11} {r['repo']:<42} "
              f"runs={r['total_runs_in_window']:<6} "
              f"failures={r['failures_in_sample']:<3} reruns={r['reruns_in_sample']}")


if __name__ == "__main__":
    main()
