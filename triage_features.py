"""
triage_features.py -- compute the model's inputs for ONE failed job, from GitHub's
list of recent runs. Same logic as `History` in build_dataset.py (a test checks they agree).

Rule that prevents leakage: only runs that FINISHED (updated_at) strictly before the
moment the job failed are counted.
"""
from datetime import datetime, timezone

import numpy as np

FAILED = ("failure", "timed_out")


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


class RunHistory:
    def __init__(self, runs):
        rows = sorted((parse_ts(r["updated_at"]).timestamp(), r.get("conclusion") in FAILED)
                      for r in runs if isinstance(r.get("updated_at"), str) and r["updated_at"])
        self.t = np.array([a for a, _ in rows], dtype=float)
        self.fail = np.concatenate([[0], np.cumsum([b for _, b in rows], dtype=int)])

    def window(self, t, hours):
        t = t.timestamp() if hasattr(t, "timestamp") else float(t)
        lo = np.searchsorted(self.t, t - hours * 3600.0, "left")
        hi = np.searchsorted(self.t, t, "left")
        return int(hi - lo), int(self.fail[hi] - self.fail[lo])


def count_features(runs, workflow_name, t, n_failed_jobs_in_run):
    """The raw counts the API expects (it computes the rates itself)."""
    allh = RunHistory(runs)
    wfh = RunHistory([r for r in runs if r.get("name") == workflow_name])
    n1, f1 = allh.window(t, 1)
    n24, f24 = allh.window(t, 24)
    n7, f7 = wfh.window(t, 24 * 7)
    return {"repo_n_1h": n1, "repo_fail_1h": f1, "repo_n_24h": n24, "repo_fail_24h": f24,
            "wf_n_7d": n7, "wf_fail_7d": f7, "n_failed_jobs_in_run": int(n_failed_jobs_in_run)}
