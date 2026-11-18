"""
check_dataset.py -- look at the dataset BEFORE modelling. Changes nothing.

Four questions:
  Q1  Is a missing log related to the label?        (if yes: 'no log' would leak)
  Q2  Which simple clues already separate the classes?
  Q3  What is going on in projects with almost no real failures?
  Q4  What would a 'dumb' model score?              (the bar we must beat)

Run:  python check_dataset.py
"""
import pandas as pd

d = pd.read_parquet("data/dataset.parquet")
lab = d[d["y"].notna()].copy()
pd.set_option("display.width", 160, "display.max_colwidth", 90)


def share(col, df=lab, min_n=30):
    g = df.groupby(col, observed=True)["y"].agg(["size", "mean"])
    g.columns = ["jobs", "share_transient"]
    return g[g["jobs"] >= min_n].round(2)


print("== Q1: log availability vs label ==")
print(lab.groupby("y")["log_status"].value_counts(normalize=True).unstack(0).round(2))
print("\nshare with an error line, by label:")
print(lab.groupby("y")["has_error_line"].mean().round(2))

print("\n== Q2: simple clues (share_transient per group) ==")
print("\nevent:\n", share("event"))
print("\nos:\n", share("os"))
print("\njob duration quartile:\n",
      share(pd.qcut(lab["job_duration_s"], 4, duplicates="drop").rename("duration")))
burst = lab["repo_fail_rate_24h"]
print("\nproject failure rate in previous 24h:\n",
      share(pd.cut(burst, [-0.01, 0.1, 0.3, 0.6, 1.0]).rename("fail_rate_24h")))
print("\nhas error line:\n", share("has_error_line"))
print("\nmetadata-type job:\n", share("is_metadata_job", min_n=1))

print("\n== Q3: projects with >= 95% transient ==")
byrepo = lab.groupby("repo")["y"].agg(["size", "mean"])
for repo in byrepo[(byrepo["mean"] >= 0.95) & (byrepo["size"] >= 10)].index:
    s = lab[lab["repo"] == repo]
    print(f"\n{repo}: {len(s)} labelled, {int(s['y'].sum())} transient")
    print(" top job names:", s["job_name"].value_counts().head(4).to_dict())
    print(" top events   :", s["event"].value_counts().head(3).to_dict())
    print(" metadata jobs:", int(s["is_metadata_job"].sum()))

print("\n== Q4: dumb baselines ==")
p = lab["y"].mean()
print(f"always 'transient': accuracy {p:.2f}, but it misses every one of the "
      f"{int((lab['y']==0).sum())} real failures")
print(f"PR-AUC of a random scorer for the 'real failure' class = {1-p:.2f}; "
      f"for the 'transient' class = {p:.2f}")
big = byrepo[byrepo["size"] >= 30]
print(f"projects with >=30 labelled jobs: {len(big)}; "
      f"with BOTH classes >=10: "
      f"{int(((lab.groupby('repo')['y'].agg(lambda s: min((s==1).sum(), (s==0).sum()))) >= 10).sum())}")
