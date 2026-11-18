"""
baselines.py  --  Step 5: first models, honest evaluation, feature-group ablation.

Question being answered: given a failed job, can we predict P(transient)?
(positive class y=1 means "transient", i.e. the rerun passed)

Two ways of testing (they answer DIFFERENT questions, so we report both):
  unseen_projects : GroupKFold by repo. Test projects were never seen in training.
                    "Does the model work on a project it has never met?"
  temporal        : train on the earlier 70% of failures (by time), test on the later 30%.
                    "Does a model trained on the past work on the future?"
                    (the same projects appear on both sides)

Models:  dummy (always the training prevalence) | logreg | random forest | hist-gradient-boosting
         (+ lightgbm if installed: pip install lightgbm)

Run:
    python baselines.py             # model comparison            (~2-5 min)
    python baselines.py --ablate    # + feature-group ablation    (a few more minutes)

Writes: results/baselines.csv, results/ablation.csv, results/oof_unseen_<model>.csv
(the out-of-fold predictions, used by the next step: calibration and cost decisions).
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

DATA = os.path.join("data", "dataset.parquet")
OUTDIR = "results"
SEED = 7

# ---- feature groups: the unit of the ablation study ---------------------------------
# Every column here is known at the moment the job fails. LEAKY columns are never used.
GROUPS = {
    "context": {"num": ["hour", "weekday", "is_pr", "is_default_branch"], "cat": ["event"]},
    "job":     {"num": ["job_duration_s"], "cat": ["os"], "text": [("job_name_norm", 3, 2000)]},
    "history": {"num": ["repo_n_1h", "repo_fail_1h", "repo_n_24h", "repo_fail_24h",
                        "repo_fail_rate_24h", "wf_n_7d", "wf_fail_7d", "wf_fail_rate_7d"]},
    "run":     {"num": ["n_failed_jobs_in_run"]},
    "log":     {"num": ["n_error_lines", "infra_hint"], "text": [("log_errors", 5, 5000)]},
}


def columns_of(groups):
    num, cat, text = [], [], []
    for g in groups:
        num += GROUPS[g].get("num", [])
        cat += GROUPS[g].get("cat", [])
        text += GROUPS[g].get("text", [])
    return num, cat, text


def make_model(name, groups, df_cols):
    """Pre-processing + model in ONE pipeline, so that everything learned from data
    (means, vocabulary, ...) is learned from the TRAINING rows only -- never the test rows."""
    num, cat, text = columns_of(groups)
    sparse_ok = name in ("logreg", "rf")        # hgb needs dense input -> no text
    if name in ("hgb", "lgbm") and text:
        text = []
    steps = []
    if num:
        parts = [("imp", SimpleImputer(strategy="median"))]
        if name == "logreg":
            parts.append(("sc", StandardScaler()))
        steps.append(("num", Pipeline(parts), num))
    if cat:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=sparse_ok), cat))
    for col, min_df, max_f in text:
        steps.append((f"txt_{col}", TfidfVectorizer(min_df=min_df, max_features=max_f,
                                                    ngram_range=(1, 2), sublinear_tf=True),
                      col))
    if not steps:
        return None
    pre = ColumnTransformer(steps, sparse_threshold=1.0 if sparse_ok else 0.0)
    clf = {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(C=1.0, max_iter=2000),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                     random_state=SEED),
        "hgb": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                              random_state=SEED),
        "lgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                               min_child_samples=20, verbose=-1, random_state=SEED)
        if LGBMClassifier else None,
    }[name]
    return None if clf is None else Pipeline([("pre", pre), ("clf", clf)])


# ---- metrics ------------------------------------------------------------------------
def ece(y, p, bins=10):
    """Expected calibration error: when the model says 70%, is it right ~70% of the time?"""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"n": len(y), "prevalence": y.mean(),
           "pr_auc_transient": average_precision_score(y, p),
           "pr_auc_real": average_precision_score(1 - y, 1 - p),
           "brier": brier_score_loss(y, p), "ece": ece(y, p)}
    out["roc_auc"] = roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan
    return out


# ---- the two evaluation schemes -----------------------------------------------------
def fit_predict(name, groups, tr, te):
    m = make_model(name, groups, tr.columns)
    if m is None:
        return None
    m.fit(tr, tr["y"].astype(int))
    return m.predict_proba(te)[:, 1]


def eval_unseen(df, name, groups, folds=5):
    """Out-of-fold predictions: every row is predicted by a model that never saw its project."""
    p = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(df, groups=df["repo"]):
        pred = fit_predict(name, groups, df.iloc[tr], df.iloc[te])
        if pred is None:
            return None, None
        p[te] = pred
    return p, metrics(df["y"], p)


def eval_temporal(df, name, groups, frac=0.7):
    d = df.sort_values("run_created_at")
    cut = int(len(d) * frac)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    pred = fit_predict(name, groups, tr, te)
    return None if pred is None else metrics(te["y"], pred)


def load_main(main_only=True, no_rerun_as_real=False):
    """The labelled rows, with a leakage check on the feature columns.

    main_only=True            labelled, non-metadata jobs (the default experiment)
    main_only=False           also keep the metadata jobs (CLA / PR-title checks)
    no_rerun_as_real=True     ALSO treat never-re-run failures as real (y=0): an
                              assumption, used only in the label-uncertainty experiment
    """
    df = pd.read_parquet(DATA)
    if no_rerun_as_real:
        m = (df["label"] == "no_rerun") & ~df["is_metadata_job"]
        df.loc[m, "y"] = 0.0
        df.loc[m, "main_set"] = True
    df = df[df["main_set"]] if main_only else df[df["y"].notna()]
    df = df.reset_index(drop=True)
    for g in GROUPS.values():                       # fail loudly if a feature is leaky
        for c in g.get("num", []) + g.get("cat", []) + [t[0] for t in g.get("text", [])]:
            assert c in df.columns, c
    leaky = {"label", "label_detail", "y", "later_attempts", "run_attempts_total",
             "sample_group", "log_status", "has_log"}
    used = {c for g in GROUPS.values() for c in g.get("num", []) + g.get("cat", [])
            + [t[0] for t in g.get("text", [])]}
    assert not (used & leaky), used & leaky
    df["infra_hint"] = df["infra_hint"].astype(int)
    df["log_errors"] = df["log_errors"].fillna("")
    df["job_name_norm"] = df["job_name_norm"].fillna("")
    return df



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true")
    args = ap.parse_args()

    df = load_main()
    print(f"{len(df)} labelled jobs, {df['repo'].nunique()} projects, "
          f"{100 * df['y'].mean():.0f}% transient\n")

    os.makedirs(OUTDIR, exist_ok=True)
    allg = list(GROUPS)
    models = ["dummy", "logreg", "rf", "hgb"] + (["lgbm"] if LGBMClassifier else [])
    rows = []
    for name in models:
        p, m = eval_unseen(df, name, allg)
        if m is None:
            continue
        rows.append({"scheme": "unseen_projects", "model": name, **m})
        pd.DataFrame({"job_id": df["job_id"], "repo": df["repo"], "y": df["y"], "p": p}
                     ).to_csv(os.path.join(OUTDIR, f"oof_unseen_{name}.csv"), index=False)
        t = eval_temporal(df, name, allg)
        rows.append({"scheme": "temporal", "model": name, **t})
        print(f"{name:<7} unseen PR-AUC(transient)={m['pr_auc_transient']:.3f} "
              f"PR-AUC(real)={m['pr_auc_real']:.3f} | temporal "
              f"PR-AUC(transient)={t['pr_auc_transient']:.3f} PR-AUC(real)={t['pr_auc_real']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, "baselines.csv"), index=False)
    print("\n" + res.round(3).to_string(index=False))

    if args.ablate:
        print("\n--- feature-group ablation (unseen projects) ---")
        ab = []
        for name in ("logreg", "hgb"):
            _, full = eval_unseen(df, name, allg)
            ab.append({"model": name, "setting": "all groups", **full})
            for g in allg:
                rest = [x for x in allg if x != g]
                for setting, groups in ((f"drop {g}", rest), (f"only {g}", [g])):
                    _, m = eval_unseen(df, name, groups)
                    if m is not None:
                        ab.append({"model": name, "setting": setting, **m})
        ab = pd.DataFrame(ab)
        ab.to_csv(os.path.join(OUTDIR, "ablation.csv"), index=False)
        print(ab[["model", "setting", "pr_auc_transient", "pr_auc_real", "roc_auc",
                  "brier"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
