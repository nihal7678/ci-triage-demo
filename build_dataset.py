"""
build_dataset.py  --  Step 4: turn the raw downloads into ONE clean table.

Input : data/raw/<owner>__<repo>/{failures.jsonl, runs.csv, done.json}
Output: data/frozen_projects.txt   the exact list of projects we use (the "freeze")
        data/dataset.parquet       one row per failed job, cleaned, with features
        data/dataset_card.json     counts + fingerprints, so the dataset is reproducible

How to run:
    python build_dataset.py

The ideas, in the order they appear in the code:
  1. FREEZE   which projects are in the dataset (only those with done.json).
  2. CLEAN    turn a raw log into normalised text the model can compare.
  3. LABEL    target + a finer label (mentor point 2: label reliability).
  4. FEATURES only facts that were known AT THE MOMENT the job failed (no leakage).
"""

import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
FROZEN = os.path.join("data", "frozen_projects.txt")
OUT = os.path.join("data", "dataset.parquet")
CARD = os.path.join("data", "dataset_card.json")
SELECTED = os.path.join("data", "selected_repos.csv")

MAX_CLEAN_CHARS = 8000     # keep the table small enough for an 8 GB laptop
MAX_ERROR_CHARS = 3000
FAILED = ("failure", "timed_out")


# ------------------------------------------------------------------ 1. FREEZE
def frozen_projects():
    """Projects that FINISHED (done.json exists). The list is written once and then
    reused, so re-running this script can never silently add a project."""
    finished = sorted(os.path.basename(os.path.dirname(p)) for p in
                      glob.glob(os.path.join(RAW, "*", "done.json")))
    if os.path.exists(FROZEN):
        with open(FROZEN, encoding="utf-8") as f:
            frozen = [x.strip() for x in f if x.strip()]
        extra = sorted(set(finished) - set(frozen))
        if extra:
            print(f"Note: {len(extra)} finished project(s) are NOT in the frozen list "
                  f"and are ignored: {extra}")
        return frozen
    with open(FROZEN, "w", encoding="utf-8") as f:
        f.write("\n".join(finished) + "\n")
    print(f"Froze {len(finished)} projects -> {FROZEN}")
    return finished


# ------------------------------------------------------------------- 2. CLEAN
TS = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ?")            # 2026-09-04T16:19:55.74Z
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")                # colour codes

# Lines every job prints, passed or failed. They carry no information.
BOILERPLATE = re.compile(
    r"(\[command\]/.*git |^adding repository directory|^removing (ssh|http|includeif)"
    r"|^cleaning up orphan|^terminate orphan|^post job cleanup|^evaluate and set job"
    r"|^set output |^cleaning up|node\.js \d+ is deprecated|^\s*$"
    r"|^http\.https://github\.com/\.extraheader|^temporarily overriding home"
    r"|^deleting the contents of|^the job was )", re.I)

# Lines that look like a problem.
ERRORISH = re.compile(
    r"(##\[error\]|\berror\b|\bfail(ed|ure|ing)?\b|exception|traceback|timed? ?out"
    r"|timeout|refused|connection reset|econnreset|denied|cannot|unable to|could not"
    r"|segmentation|killed|out of memory|no space left|panic|assert)", re.I)

# Lines that contain an error-ish word but are NOT errors (test-tool headers, compiler
# warnings, GitHub's own step bookkeeping). Found by reading the most common "error"
# lines in the sample files -- re-check this list on the full data.
NOISE = re.compile(r"(timeout method|timeout func_only|^plugins:|timeout: ?\d|##\[end-action"
                   r"|##\[group\]|: warning:|^=+ .*=+$)", re.I)

# Hints that the cause was the environment, not the code (heuristic!).
INFRA = re.compile(
    r"(econnreset|connection reset|timed out|connection timeout|read timeout|temporary failure|could not resolve"
    r"|\b50[234]\b|rate limit|too many requests|no space left|runner has received a shutdown"
    r"|lost communication|network is unreachable|tls handshake|download failed"
    r"|failed to download|service unavailable|socket hang up|eai_again)", re.I)

HEX = re.compile(r"\b[0-9a-f]{7,64}\b")
URL = re.compile(r"https?://\S+")
PATH = re.compile(r"(?:[A-Za-z]:\\|/)(?:[\w.\-@]+[/\\])+[\w.\-@]*")
NUM = re.compile(r"\d+(?:\.\d+)*")


def normalise(line):
    """Make two lines that differ only by an id/number/path look identical."""
    s = line.lower()
    s = URL.sub(" <url> ", s)
    s = PATH.sub(" <path> ", s)
    s = HEX.sub(" <hash> ", s)
    s = NUM.sub("<n>", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_log(raw):
    """Return (cleaned text, error-looking lines only, raw-error-text for hints)."""
    kept, errs, raw_errs = [], [], []
    for line in (raw or "").splitlines():
        line = TS.sub("", ANSI.sub("", line)).rstrip()
        if BOILERPLATE.search(line.strip()):
            continue
        n = normalise(line)
        if not n:
            continue
        kept.append(n)
        if ERRORISH.search(line) and not NOISE.search(line.strip()):
            errs.append(n)
            raw_errs.append(line)
    return ("\n".join(kept)[-MAX_CLEAN_CHARS:],
            "\n".join(errs)[-MAX_ERROR_CHARS:],
            "\n".join(raw_errs))


# -------------------------------------------------------------------- 3. LABEL
# Jobs whose outcome depends on the PR text/metadata, not on the code. A "pass on
# rerun" here usually means someone edited the PR title, not that the job flaked.
METADATA_JOB = re.compile(
    r"(title|semantic|conventional|lint-?pr|labeler|\blabel|\bcla\b|license/cla|link ?check"
    r"|linkcheck|dependency-review|stale|changelog|pr-?check|check-?pr|assign|triage)", re.I)


def label_detail(label, infra_hint, is_meta):
    """Finer label (mentor point 2). It is a HEURISTIC, which is exactly why we later
    measure how much the results change if we drop the uncertain classes."""
    if label == "passed_on_rerun":
        if is_meta:
            return "metadata_pass"          # suspicious: probably not a real flake
        return "transient_infra" if infra_hint else "transient_other"
    if label == "failed_on_rerun":
        return "persistent"
    return "ambiguous" if label == "ambiguous" else "unknown"


# ----------------------------------------------------------------- 4. FEATURES
def ts(series):
    return pd.to_datetime(series, utc=True, errors="coerce")


def os_hint(text):
    t = text.lower()
    for key, name in (("windows", "windows"), ("win", "windows"), ("macos", "macos"),
                      ("osx", "macos"), ("mac", "macos"), ("ubuntu", "linux"),
                      ("linux", "linux")):
        if key in t:
            return name
    return "unknown"


class History:
    """Failure-rate in a window BEFORE a moment t, from the project's runs.csv.

    Trick: sort runs by finish time, keep a running total of failures. Then 'how many
    runs / failures finished in [t-w, t)' is two binary searches and a subtraction,
    instead of scanning every run for every failure.

    No-leakage rule: only runs that FINISHED (updated_at) before t are counted. A run
    that was re-run later has a later updated_at, so it is simply not counted yet --
    we never see its future final result."""

    def __init__(self, runs):
        runs = runs.dropna(subset=["updated_at"]).sort_values("updated_at")
        self.t = runs["updated_at"].values.astype("datetime64[ns]").astype("int64")
        self.fail = np.concatenate([[0], np.cumsum(runs["conclusion"].isin(FAILED).values)])

    def window(self, t, hours):
        t = np.int64(pd.Timestamp(t).value)
        lo = np.searchsorted(self.t, t - int(hours * 3.6e12), "left")
        hi = np.searchsorted(self.t, t, "left")
        return hi - lo, int(self.fail[hi] - self.fail[lo])


def load_project(folder, language):
    repo = os.path.basename(folder).replace("__", "/", 1)
    fpath = os.path.join(folder, "failures.jsonl")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    with open(fpath, encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    if df.empty:
        return df
    df["language"] = language
    df = df.drop_duplicates(subset=["repo", "job_id"])
    df["n_failed_jobs_in_run"] = df.groupby("run_id")["job_id"].transform("size")

    runs = pd.read_csv(os.path.join(folder, "runs.csv"))
    runs["updated_at"] = ts(runs["updated_at"])
    hist_all = History(runs)
    by_wf = {w: History(g) for w, g in runs.groupby("name")}

    when = ts(df["job_completed_at"]).fillna(ts(df["run_created_at"]))
    cols = {k: [] for k in ("repo_n_1h", "repo_fail_1h", "repo_n_24h", "repo_fail_24h",
                            "wf_n_7d", "wf_fail_7d")}
    for t, wf in zip(when, df["workflow"]):
        n1, f1 = hist_all.window(t, 1)
        n24, f24 = hist_all.window(t, 24)
        n7, f7 = by_wf[wf].window(t, 24 * 7) if wf in by_wf else (0, 0)
        for k, v in zip(cols, (n1, f1, n24, f24, n7, f7)):
            cols[k].append(v)
    for k, v in cols.items():
        df[k] = v
    return df


def build_features(df):
    cleaned = df["log_tail"].fillna("").map(clean_log)
    df["log_clean"] = [c[0] for c in cleaned]
    df["log_errors"] = [c[1] for c in cleaned]
    infra = [bool(INFRA.search(c[2])) for c in cleaned]

    df["has_log"] = (df["log_status"] == "ok") & (df["log_clean"].str.len() > 0)
    df["n_error_lines"] = df["log_errors"].map(lambda s: len(s.splitlines()) if s else 0)
    df["has_error_line"] = df["n_error_lines"] > 0
    df["infra_hint"] = infra

    df["is_metadata_job"] = (df["job_name"].fillna("") + " " + df["workflow"].fillna("")
                             ).map(lambda s: bool(METADATA_JOB.search(s)))
    df["label_detail"] = [label_detail(l, i, m) for l, i, m in
                          zip(df["label"], df["infra_hint"], df["is_metadata_job"])]
    # target: 1 = transient (rerun passed), 0 = persistent (rerun failed), NaN = not usable
    df["y"] = df["label"].map({"passed_on_rerun": 1.0, "failed_on_rerun": 0.0})

    # main experiment = usable label AND not a metadata job (PR-title / CLA / link checks)
    df["main_set"] = df["y"].notna() & ~df["is_metadata_job"]

    # --- features known at failure time ---
    start, end = ts(df["job_started_at"]), ts(df["job_completed_at"])
    df["job_duration_s"] = (end - start).dt.total_seconds().clip(lower=0)
    created = ts(df["run_created_at"])
    df["hour"] = created.dt.hour
    df["weekday"] = created.dt.day_of_week
    df["is_pr"] = (df["event"] == "pull_request").astype(int)
    df["is_default_branch"] = df["head_branch"].isin(["main", "master", "develop", "dev"]).astype(int)
    df["job_name_norm"] = df["job_name"].fillna("").map(normalise)
    df["os"] = (df["job_name"].fillna("") + " " + df["log_clean"].str[:300]).map(os_hint)
    df["repo_fail_rate_24h"] = df["repo_fail_24h"] / df["repo_n_24h"].clip(lower=1)
    df["wf_fail_rate_7d"] = df["wf_fail_7d"] / df["wf_n_7d"].clip(lower=1)
    df["run_created_at"] = created
    return df


# Columns that must NEVER be used as model inputs, because they contain the future.
# log_status / has_log describe OUR collector (3-logs-per-run cap), not the failure.
LEAKY = ["log_status", "has_log", "label", "label_detail", "y", "later_attempts", "run_attempts_total", "sample_group"]


def main():
    projects = frozen_projects()
    if not projects:
        sys.exit("No finished projects found in data/raw.")
    lang = {}
    if os.path.exists(SELECTED):
        sel = pd.read_csv(SELECTED)
        lang = {r.replace("/", "__"): l for r, l in zip(sel["repo"], sel["language"])}

    parts = []
    for p in projects:
        d = load_project(os.path.join(RAW, p), lang.get(p, "unknown"))
        if not d.empty:
            parts.append(d)
            print(f"  {p:<40} {len(d):>5} failed jobs")
    df = build_features(pd.concat(parts, ignore_index=True))

    keep = ["repo", "language", "run_id", "job_id", "workflow", "job_name", "job_name_norm",
            "event", "head_branch", "run_created_at", "log_status", "has_log", "log_clean",
            "log_errors", "n_error_lines", "has_error_line", "infra_hint", "is_metadata_job",
            "os", "hour", "weekday", "is_pr", "is_default_branch", "job_duration_s",
            "repo_n_1h", "repo_fail_1h", "repo_n_24h", "repo_fail_24h", "repo_fail_rate_24h",
            "wf_n_7d", "wf_fail_7d", "wf_fail_rate_7d", "n_failed_jobs_in_run", "main_set"] + LEAKY
    df = df[list(dict.fromkeys(keep))]   # LEAKY columns overlap with keep: no duplicates
    df.to_parquet(OUT, index=False)

    # ---- fingerprint of the raw files: if they change, the card changes ----
    h = hashlib.sha256()
    for p in projects:
        with open(os.path.join(RAW, p, "failures.jsonl"), "rb") as f:
            h.update(f.read())
    card = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "projects": projects, "n_projects": len(projects), "rows": len(df),
            "label_counts": df["label"].value_counts().to_dict(),
            "label_detail_counts": df["label_detail"].value_counts().to_dict(),
            "first_run": str(df["run_created_at"].min()), "last_run": str(df["run_created_at"].max()),
            "raw_sha256": h.hexdigest(), "leaky_columns_never_use_as_input": LEAKY}
    with open(CARD, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    # ---- the report you should READ, not skim ----
    lab = df[df["y"].notna()]
    print(f"\n{len(df)} failed jobs from {len(projects)} projects -> {OUT}")
    print("\nlabels:\n" + df["label"].value_counts().to_string())
    print("\nfiner labels:\n" + df["label_detail"].value_counts().to_string())
    print(f"\nlabelled rows (usable for training): {len(lab)}  "
          f"({100 * lab['y'].mean():.0f}% transient)")
    print(f"rows with a log                 : {100 * df['has_log'].mean():.0f}%")
    print(f"rows with at least one error line: {100 * df['has_error_line'].mean():.0f}%")
    g = lab.groupby("repo")["y"].agg(["size", "mean"]).sort_values("size", ascending=False)
    g.columns = ["labelled_jobs", "share_transient"]
    print("\nper project (labelled only):\n" + g.round(2).to_string())


if __name__ == "__main__":
    main()
