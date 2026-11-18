"""
triage_model.py -- the small pieces shared by TRAINING and SERVING.

Keeping them in one file matters: if training and serving each had their own copy,
they could drift apart (this is called train/serve skew) and the model would quietly
give wrong answers in production.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

# The columns the model was trained on (feature groups "history" + "run").
FEATURES = ["repo_n_1h", "repo_fail_1h", "repo_n_24h", "repo_fail_24h", "repo_fail_rate_24h",
            "wf_n_7d", "wf_fail_7d", "wf_fail_rate_7d", "n_failed_jobs_in_run"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Platt:
    """Calibration: turns the model's raw score into an honest probability.
    (a one-variable logistic regression on the logit of the raw score)"""

    def fit(self, scores, y):
        self.m = LogisticRegression(C=1e6).fit(logit(np.asarray(scores)).reshape(-1, 1), y)
        return self

    def predict(self, scores):
        return self.m.predict_proba(logit(np.asarray(scores)).reshape(-1, 1))[:, 1]


def decision_threshold(job_minutes, X=15.0, W=1.0):
    """Re-run when  p*X > (1-p)*(1+W)*d   <=>   p > (1+W)d / (X + (1+W)d).
    X = cost of a human stepping in, W = weight of the developer's waiting,
    d = how long the failed job ran (minutes)."""
    d = max(float(job_minutes), 0.0)
    return (1 + W) * d / (X + (1 + W) * d)


def decide(p_flaky, job_minutes, X=15.0, W=1.0):
    return "rerun" if p_flaky > decision_threshold(job_minutes, X, W) else "human"
