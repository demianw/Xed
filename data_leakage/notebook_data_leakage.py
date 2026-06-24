# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
import os
if not os.path.exists("Xed"):
    os.system("git clone --depth=1 https://github.com/demianw/Xed.git")
if "data_leakage" not in os.getcwd():
    os.chdir("Xed/data_leakage")

# %%
# %pip install -q scikit-learn==1.6.1 pandas==2.2.3 matplotlib==3.9.0 numpy==1.26.4

# %% [markdown]
# # Data Leakage through Incorrect Cross-Validation
#
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Define *data leakage* and explain why it produces over-optimistic performance estimates.
# 2. Identify the two most common leakage patterns in cross-validation:
#    (a) preprocessing fitted on all data before the CV loop, and
#    (b) feature selection driven by the target before the CV loop.
# 3. Demonstrate the bias quantitatively using a controlled simulation.
# 4. Fix both patterns by encapsulating every transformation inside a scikit-learn
#    `Pipeline` so that fitting happens exclusively on each training fold.
# 5. Detect a third leakage pattern specific to time-series data.

# %% [markdown]
# ---
# ## 1. What is data leakage?
#
# **Data leakage** occurs when information from outside the *legitimate* training set
# is used to build or evaluate a model.  The result is an estimate of generalisation
# performance that looks better than it really is — sometimes dramatically so.
#
# The most common source of leakage in practice is deceptively simple:
# a preprocessing step (e.g. scaling, imputation, feature selection) is fitted on
# the *whole* dataset and *then* cross-validation is run.  This means that each
# test fold has already influenced the preprocessing, violating the assumption that
# the test fold is "unseen".
#
# ```
#  WRONG                            CORRECT
#  ─────────────────────────────    ────────────────────────────────────
#  scaler.fit_transform(X_all)  →   Pipeline([scaler, model]).fit(X_train)
#  for train, test in cv:                  ↑ fitted fresh on each fold
#      model.fit(X_scaled[train])
#      model.score(X_scaled[test])  ← test fold already "saw" scaler
# ```
#
# The leakage is subtle because each test fold is *not* used in `model.fit()`.
# But it *is* used in `scaler.fit_transform()`, which sets the mean and variance
# that normalise every subsequent fold.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance

rng = np.random.RandomState(42)
%matplotlib inline

# %% [markdown]
# ---
# ## 2. Leakage from preprocessing outside the CV loop
#
# ### 2.1 A concrete example: scaling before cross-validation
#
# We use the breast-cancer dataset, which has 30 numerical features on very
# different scales (some in mm, some as raw counts).

# %%
X, y = load_breast_cancer(return_X_y=True)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
      f"{y.mean()*100:.1f}% positive class")
print(f"Feature ranges (min/max of column-wise min/max):")
print(f"  smallest feature range: {X.min(axis=0).min():.4f} – {X.max(axis=0).max():.4f}")

# %% [markdown]
# #### The wrong way — scale *before* CV

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=10_000, random_state=42)

# ── WRONG: scaler sees ALL data including every test fold ──────────────────
scaler_global = StandardScaler()
X_scaled_global = scaler_global.fit_transform(X)          # leakage here

scores_wrong = cross_val_score(model, X_scaled_global, y, cv=cv, scoring="accuracy")
print(f"Wrong (global scaling) — mean accuracy: {scores_wrong.mean():.4f}  "
      f"± {scores_wrong.std():.4f}")

# %% [markdown]
# #### The correct way — scale *inside* the pipeline

# %%
# ── CORRECT: scaler is re-fitted on each training fold only ───────────────
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10_000, random_state=42))

scores_correct = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print(f"Correct (pipeline) — mean accuracy: {scores_correct.mean():.4f}  "
      f"± {scores_correct.std():.4f}")

# %% [markdown]
# ### Question 1
#
# Run the two cells above and compare the mean accuracy values.
# Is the difference large or small on this dataset?
# Would the gap be larger or smaller if the training set were much smaller?
# Try re-running with `n_splits=10` and observe the variances.

# %%

# %% [markdown]
# ---
# ## 3. A catastrophic leakage example: feature selection outside CV
#
# Scaling leakage is subtle. Feature-selection leakage can be *catastrophic*.
# The following simulation shows how a model can appear to have near-perfect
# accuracy on **completely random data** if feature selection is done outside CV.
#
# ### 3.1 The simulation setup
#
# We generate a dataset where **the target is pure random noise** — there is
# genuinely no signal in the features.  A correctly evaluated classifier should
# score at chance level (≈50%).

# %%
n_samples = 100
n_features = 10_000   # many more features than samples

# Pure noise: neither X nor y contain any real signal
X_noise = rng.randn(n_samples, n_features)
y_noise = rng.randint(0, 2, n_samples)   # random binary labels

print(f"X shape: {X_noise.shape}")
print(f"Class balance: {y_noise.mean()*100:.0f}% positive")

# %% [markdown]
# ### 3.2 The wrong way — select features on the full dataset

# %%
# ── WRONG: SelectKBest sees all labels, including those in every test fold ─
selector_global = SelectKBest(f_classif, k=50)
X_selected_global = selector_global.fit_transform(X_noise, y_noise)  # leakage!

cv_noise = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_noise = LogisticRegression(max_iter=1000, random_state=42)

scores_fs_wrong = cross_val_score(
    model_noise, X_selected_global, y_noise, cv=cv_noise, scoring="accuracy"
)
print(f"Wrong (global feature selection) — accuracy: {scores_fs_wrong.mean():.4f}  "
      f"± {scores_fs_wrong.std():.4f}")
print(">>> This is PURE NOISE data — any accuracy above 0.50 is an artefact!")

# %% [markdown]
# ### 3.3 The correct way — feature selection inside the pipeline

# %%
# ── CORRECT: selector is re-fitted on each training fold ──────────────────
pipe_fs_correct = make_pipeline(
    SelectKBest(f_classif, k=50),
    LogisticRegression(max_iter=1000, random_state=42)
)

scores_fs_correct = cross_val_score(
    pipe_fs_correct, X_noise, y_noise, cv=cv_noise, scoring="accuracy"
)
print(f"Correct (pipeline) — accuracy: {scores_fs_correct.mean():.4f}  "
      f"± {scores_fs_correct.std():.4f}")
print(">>> Now we correctly recover chance-level performance.")

# %% [markdown]
# ### 3.4 Visualising the bias

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(
    ["Wrong\n(global selection)", "Correct\n(pipeline)"],
    [scores_fs_wrong.mean(), scores_fs_correct.mean()],
    yerr=[scores_fs_wrong.std(), scores_fs_correct.std()],
    color=["#d62728", "#2ca02c"], capsize=8, width=0.4
)
ax.axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance level (50%)")
ax.set_ylim(0, 1.05)
ax.set_ylabel("CV accuracy")
ax.set_title("Leakage from feature selection on pure-noise data")
ax.legend()
plt.tight_layout()
plt.show()
print(f"\nBias introduced by leakage: "
      f"{(scores_fs_wrong.mean() - scores_fs_correct.mean())*100:.1f} percentage points")

# %% [markdown]
# ### Question 2
#
# The magnitude of the leakage bias depends on the ratio `n_features / n_samples`.
#
# **2a.** Re-run sections 3.1–3.4 with `n_features = 500` instead of 10 000.
# How large is the bias now?  Why does reducing the number of features reduce
# the leakage?
#
# **2b.** Re-run with `n_features = 10_000` but `k = 5` (select only 5 features
# instead of 50).  How does the number of selected features interact with the bias?
#
# **2c.** Can you derive an intuitive explanation for when feature-selection
# leakage is most dangerous?

# %%

# %% [markdown]
# ---
# ## 4. How leakage misleads model selection
#
# Leakage does not just inflate a single score — it can cause you to *prefer
# the wrong model* when comparing multiple methods.
#
# ### Question 3
#
# Below we set up a genuine regression problem (from `make_regression`) where the
# true signal lives in 20 out of 100 features.
#
# For each of the three pipelines below, compute 5-fold CV `r2` scores:
#
# 1. **Leaky pipeline**: `StandardScaler().fit_transform(X_all)` → `Ridge()` with CV
# 2. **Correct pipeline A**: `make_pipeline(StandardScaler(), Ridge())`
# 3. **Correct pipeline B**: `make_pipeline(StandardScaler(), SelectKBest(k=20), Ridge())`
#
# Which pipeline *appears* best under the leaky evaluation?  Which *is* actually best
# under correct evaluation?  Comment on what this implies for model comparison.

# %%
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(
    n_samples=200, n_features=100, n_informative=20,
    noise=20, random_state=42
)

cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline 1 — leaky baseline (global scaling before CV)
scaler_reg = StandardScaler()
X_reg_scaled_global = scaler_reg.fit_transform(X_reg)
scores_leaky = cross_val_score(Ridge(), X_reg_scaled_global, y_reg, cv=cv_reg, scoring="r2")

# Pipeline 2 — correct: scaling inside pipeline
pipe_ridge = make_pipeline(StandardScaler(), Ridge())
scores_pipe_a = cross_val_score(pipe_ridge, X_reg, y_reg, cv=cv_reg, scoring="r2")

# Pipeline 3 — correct: scaling + feature selection inside pipeline
pipe_ridge_fs = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), Ridge())

# ── your code here ──
# scores_pipe_b = ...

# %% [markdown]
# ---
# ## 5. Time-series leakage: using future data to predict the past
#
# In time-ordered data there is a third leakage pattern: using a random CV split
# means that some training examples are *later in time* than some test examples.
# The model then effectively learns from "future" data when predicting the "past".
#
# ### 5.1 Simulated autocorrelated time series

# %%
from sklearn.model_selection import TimeSeriesSplit

# Simulate a slowly drifting signal (strong autocorrelation)
n_time = 300
t = np.arange(n_time)
signal = np.sin(0.05 * t) + 0.3 * rng.randn(n_time)

# Features: lags 1–5
lag_features = np.stack([signal[i: n_time - 5 + i] for i in range(5)], axis=1)
target_ts = signal[5:]           # predict one step ahead

print(f"Time-series features: {lag_features.shape}, target: {target_ts.shape}")

# %% [markdown]
# ### 5.2 Standard KFold vs TimeSeriesSplit

# %%
from sklearn.linear_model import LinearRegression

model_ts = make_pipeline(StandardScaler(), LinearRegression())

# ── WRONG: random folds ignore temporal order ──────────────────────────────
cv_random = KFold(n_splits=5, shuffle=True, random_state=42)
scores_random = cross_val_score(model_ts, lag_features, target_ts,
                                cv=cv_random, scoring="r2")

# ── CORRECT: test folds are always in the future relative to training folds ─
cv_ts = TimeSeriesSplit(n_splits=5)
scores_ts = cross_val_score(model_ts, lag_features, target_ts,
                            cv=cv_ts, scoring="r2")

print(f"Random KFold  R²: {scores_random.mean():.4f} ± {scores_random.std():.4f}")
print(f"TimeSeriesSplit R²: {scores_ts.mean():.4f} ± {scores_ts.std():.4f}")

# %% [markdown]
# ### 5.3 Visualising the split structure

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

for ax, (cv_obj, title) in zip(axes, [
    (cv_random, "KFold (WRONG for time series)"),
    (cv_ts,     "TimeSeriesSplit (CORRECT)")
]):
    for fold, (train_idx, test_idx) in enumerate(cv_obj.split(lag_features)):
        ax.scatter(train_idx, [fold] * len(train_idx),
                   marker="|", color="#2ca02c", alpha=0.4, s=10)
        ax.scatter(test_idx,  [fold] * len(test_idx),
                   marker="|", color="#d62728", alpha=0.8, s=10)
    ax.set_ylabel("Fold")
    ax.set_title(title)

axes[-1].set_xlabel("Time index")
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#2ca02c", lw=3, label="Train"),
    Line2D([0], [0], color="#d62728", lw=3, label="Test"),
]
axes[0].legend(handles=legend_elements, loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Question 4
#
# **4a.** In the visualisation above, which of the two strategies allows
# the model to train on data from the *future* relative to the test set?
# Shade one such overlap on the figure in your mind and explain in words
# why this constitutes leakage.
#
# **4b.** Replace `LinearRegression` with `Ridge` and repeat the comparison.
# Does regularisation reduce the gap between the two CV estimates?

# %%

# %% [markdown]
# ---
# ## 6. Leakage checklist
#
# Before trusting any cross-validated performance estimate, go through this
# checklist:
#
# | # | Question | Correct answer |
# |---|----------|----------------|
# | 1 | Is every preprocessing step (scaling, imputation, encoding, feature selection) fitted **only** on training folds? | Yes — use a `Pipeline` |
# | 2 | Is any statistic derived from the target (mean encoding, SMOTE, etc.) computed **before** the fold split? | No — put it inside the pipeline |
# | 3 | For time-series or grouped data, do test folds come **after** (or are disjoint from) training folds? | Yes — use `TimeSeriesSplit` or `GroupKFold` |
# | 4 | Are you preprocessing train and test separately in a manual loop? | Always use `Pipeline` |
#
# **The golden rule:** if your preprocessing object has a `.fit()` method,
# it belongs inside a `Pipeline`.

# %% [markdown]
# ---
# ## 7. Summary exercises
#
# ### Exercise A — Diagnose the bug
#
# A colleague shares the following code and reports 95% accuracy on a dataset
# with 50 samples and 5 000 features.  Identify every leakage problem and rewrite
# the code correctly.
#
# ```python
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
#
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)
#
# pca = PCA(n_components=10)
# X_pca = pca.fit_transform(X_std)
#
# selector = SelectKBest(f_classif, k=5)
# X_sel = selector.fit_transform(X_pca, y)
#
# scores = cross_val_score(SVC(), X_sel, y, cv=5)
# print(scores.mean())
# ```

# %%
# Paste your corrected version here

# %% [markdown]
# ### Exercise B — Quantify the damage
#
# Using the breast-cancer dataset, build a grid of experiments:
#
# - Methods: `[StandardScaler, PCA(n_components=5), SelectKBest(k=5)]`
#   (all combinations: 0, 1, 2, or 3 steps applied globally before CV)
# - For each combination, compute:
#   - the **leaky** CV accuracy (transformers fitted globally)
#   - the **correct** CV accuracy (transformers inside a pipeline)
# - Plot the leaky vs. correct accuracy for every combination as a scatter plot.
#
# What pattern do you observe?  Which transformer introduces the most leakage?

# %%

# %% [markdown]
# ---
# ## Summary
#
# In this notebook you:
# - Defined data leakage as information flowing from the test set into the model
#   fitting process, producing inflated performance estimates.
# - Showed that **scaling before CV** introduces a small but non-zero bias.
# - Demonstrated that **feature selection before CV** can inflate accuracy from
#   50 % to well above 80 % on pure-noise data (catastrophic leakage).
# - Established that the bias grows with the ratio `n_features / n_samples` and
#   the number of selected features.
# - Identified **temporal leakage** as a third pattern, corrected by `TimeSeriesSplit`.
# - Summarised the golden rule: every `.fit()` call belongs inside a `Pipeline`.
#
# **Next notebook:** *Cross-Validation and Scoring Methods* — now that you know what
# can go wrong, that notebook teaches you the correct CV machinery in depth.
