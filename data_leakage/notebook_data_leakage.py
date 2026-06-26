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

# %%
%pip install -q git+https://github.com/demianw/Xed.git

# %%

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
# Any preprocessing step that has a ``.fit()`` method can introduce leakage
# when fitted on the whole dataset before cross-validation.
# The severity of the bias depends entirely on **how much the preprocessing
# step uses the target variable**.
#
# | Preprocessing step | Leaks target? | Typical bias |
# |--------------------|---------------|--------------|
# | `StandardScaler` | No — fits on features only | Near-zero in practice |
# | `SimpleImputer(strategy='mean')` | No | Near-zero |
# | Mean encoding (group-mean of target) | **Yes — directly** | Large, clearly visible |
# | `SelectKBest(f_classif)` | **Yes — uses labels** | Catastrophic (Section 3) |
#
# #### Why `SelectKBest(f_classif)` leaks
#
# `f_classif` computes ANOVA F-statistics using **all** the data, including the
# test fold's labels.  The feature selection "sees" the test labels and picks
# features that correlate with them — even if that correlation is spurious.
# When the selector is fitted globally before CV, every test fold has already
# influenced which features survive, so the downstream classifier is scored on
# a feature set that was *chosen using the answers*.  The result is a
# catastrophic inflation of accuracy, as Section 3 demonstrates on pure-noise
# data where no real signal exists.
#
# ### 2.1 Why StandardScaler leakage is nearly invisible
#
# `StandardScaler` leakage is theoretically real: the test fold's feature values
# shift the global mean and variance slightly compared to the training-fold-only values.
# In practice on a moderate dataset this shift is **~0.0001%** — far too small to
# change any prediction.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

X_bc, y_bc = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_lr = LogisticRegression(max_iter=10_000, random_state=42)

# ── WRONG ──────────────────────────────────────────────────────────────────────
scaler_global = StandardScaler()
X_scaled_global = scaler_global.fit_transform(X_bc)  # test fold shifts the mean
scores_wrong_scaler = cross_val_score(model_lr, X_scaled_global, y_bc, cv=cv, scoring="accuracy")

# ── CORRECT ────────────────────────────────────────────────────────────────────
pipe_scaler = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=10_000, random_state=42),
)
scores_correct_scaler = cross_val_score(pipe_scaler, X_bc, y_bc, cv=cv, scoring="accuracy")

print("StandardScaler leakage on breast-cancer (569 samples, 5 folds)")
print(f"  Wrong (global scaling) : {scores_wrong_scaler.mean():.4f}")
print(f"  Correct (pipeline)     : {scores_correct_scaler.mean():.4f}")
print(
    f"  Difference             : {scores_wrong_scaler.mean() - scores_correct_scaler.mean():+.4f}"
)
print()
print("  The scaler statistics shift by only ~0.0001% when excluding a 20%-fold.")
print("  Always use Pipeline anyway: the cost is zero, and correctness is a principle.")

# %% [markdown]
# ### 2.2 Mean encoding leakage — a common, visible mistake
#
# **Mean encoding** (also called *target encoding*) replaces a categorical feature
# with the mean of the target variable within that category.  It is popular in
# insurance because features like *region* or *vehicle brand* have many levels and
# high predictive power when encoded by average claim rate.
#
# When mean encoding is fitted on **all data before CV**, the test fold's target
# values are used to compute the category means — a direct feed of the answer
# into the features the model will be scored on.  With many categories and a
# small dataset, this inflates performance substantially.
#
# **Insurance context:**  Imagine computing "mean claim frequency per region"
# from the whole dataset, then using it as a feature.  The model learns
# "region R24 has 9.2% frequency" — but that 9.2% was computed partly from
# the claims in the test set.  In production the model would never have seen
# those future claims, so the learned statistic is optimistic.

# %%
import pandas as pd

rng_enc = np.random.default_rng(42)

# Simulate an insurance-style dataset:
#   500 policyholders, 20 regions, 5 noise features, binary claim outcome.
n_enc, n_regions = 500, 20
region_arr = rng_enc.integers(0, n_regions, n_enc)
noise_arr = rng_enc.normal(size=(n_enc, 5))
region_risk = rng_enc.uniform(0.1, 0.9, n_regions)  # true P(claim) per region
y_enc = rng_enc.binomial(1, region_risk[region_arr]).astype(float)

print(
    f"Dataset: {n_enc} policies, {n_regions} regions, "
    f"overall claim rate = {y_enc.mean() * 100:.1f}%"
)

# ── WRONG: compute region mean using all rows, including the test fold ──────
df_enc = pd.DataFrame(noise_arr, columns=[f"noise_{i}" for i in range(5)])
df_enc["region"] = region_arr
df_enc["y"] = y_enc

global_region_mean = df_enc.groupby("region")["y"].transform("mean")  # leakage!
X_wrong_enc = np.column_stack([noise_arr, global_region_mean.values])

cv_enc = KFold(n_splits=5, shuffle=True, random_state=42)
scores_wrong_enc = cross_val_score(
    LogisticRegression(max_iter=500, random_state=42),
    X_wrong_enc,
    y_enc,
    cv=cv_enc,
    scoring="accuracy",
)


# ── CORRECT: compute region mean only from the training fold ────────────────
def mean_encode_fold(X_df, y_arr, train_idx, test_idx, cat_col, target_col, n_cats):
    """Compute target mean per category using training fold only."""
    X_tr, y_tr = X_df.iloc[train_idx], y_arr[train_idx]
    X_te = X_df.iloc[test_idx]
    fold_means = y_tr.groupby(X_tr[cat_col]).mean()
    global_mean = y_tr.mean()  # fallback for unseen categories
    enc_tr = X_tr[cat_col].map(fold_means).fillna(global_mean).values
    enc_te = X_te[cat_col].map(fold_means).fillna(global_mean).values
    noise_cols = [c for c in X_df.columns if c not in (cat_col, target_col)]
    Xf_tr = np.column_stack([X_tr[noise_cols].values, enc_tr])
    Xf_te = np.column_stack([X_te[noise_cols].values, enc_te])
    return Xf_tr, y_tr.values, Xf_te, y_arr[test_idx]


scores_correct_enc = []
for tr_idx, te_idx in cv_enc.split(df_enc, y_enc):
    Xf_tr, yf_tr, Xf_te, yf_te = mean_encode_fold(
        df_enc,
        pd.Series(y_enc),
        tr_idx,
        te_idx,
        cat_col="region",
        target_col="y",
        n_cats=n_regions,
    )
    m = LogisticRegression(max_iter=500, random_state=42)
    m.fit(Xf_tr, yf_tr)
    scores_correct_enc.append(m.score(Xf_te, yf_te))
scores_correct_enc = np.array(scores_correct_enc)

print()
print("Mean encoding leakage on simulated insurance data")
print(f"  Wrong (global encoding)  : {scores_wrong_enc.mean():.4f} ± {scores_wrong_enc.std():.4f}")
print(
    f"  Correct (fold encoding)  : {scores_correct_enc.mean():.4f} ± {scores_correct_enc.std():.4f}"
)
print(f"  Leakage bias             : {scores_wrong_enc.mean() - scores_correct_enc.mean():+.4f}")

# %% [markdown]
# ### 2.3 The pure-noise proof
#
# The clearest possible demonstration: assign **completely random labels** to the
# policyholders, so the correct CV accuracy should be ~50%.
# Global mean encoding still finds "structure" because the category means
# memorise the (random) test labels — and the model learns from that noise.

# %%
y_noise_enc = rng_enc.integers(0, 2, n_enc).astype(float)  # completely random labels

global_mean_noise = df_enc.assign(y=y_noise_enc).groupby("region")["y"].transform("mean")
X_noise_wrong = np.column_stack([noise_arr, global_mean_noise.values])
scores_noise_wrong = cross_val_score(
    LogisticRegression(max_iter=500, random_state=42),
    X_noise_wrong,
    y_noise_enc,
    cv=cv_enc,
    scoring="accuracy",
)

df_noise = df_enc.copy()
df_noise["y"] = y_noise_enc
scores_noise_correct = []
for tr_idx, te_idx in cv_enc.split(df_noise, y_noise_enc):
    Xf_tr, yf_tr, Xf_te, yf_te = mean_encode_fold(
        df_noise,
        pd.Series(y_noise_enc),
        tr_idx,
        te_idx,
        cat_col="region",
        target_col="y",
        n_cats=n_regions,
    )
    m = LogisticRegression(max_iter=500, random_state=42)
    m.fit(Xf_tr, yf_tr)
    scores_noise_correct.append(m.score(Xf_te, yf_te))
scores_noise_correct = np.array(scores_noise_correct)

print("Mean encoding leakage on PURE NOISE labels (correct answer = 50%)")
print(f"  Wrong (global encoding)  : {scores_noise_wrong.mean():.4f} ← should be 0.50!")
print(f"  Correct (fold encoding)  : {scores_noise_correct.mean():.4f} ← back to chance")
print(
    f"  Leakage bias             : {scores_noise_wrong.mean() - scores_noise_correct.mean():+.4f}"
)

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 1 — Mean encoding leakage</b>
# <ul>
#   <li>
#     <b>Vary the number of regions.</b>
#     Re-run the mean encoding experiment (with real labels) for
#     <code>n_regions ∈ [5, 10, 20, 50, 100]</code>, keeping <code>n_enc = 500</code>.
#     How does the leakage bias change as the number of regions grows?
#     Why does a larger number of categories make leakage worse?
#   </li>
#   <li>
#     <b>Sklearn fix.</b>
#     <code>sklearn.preprocessing.TargetEncoder</code> (available since sklearn 1.3)
#     implements fold-safe target encoding using internal cross-fitting.
#     Replace the manual <code>mean_encode_fold</code> loop with a pipeline:
#     <pre>
# from sklearn.preprocessing import TargetEncoder
# pipe = make_pipeline(TargetEncoder(random_state=42), LogisticRegression(...))
# scores = cross_val_score(pipe, X_cat_only, y_enc, cv=cv_enc, scoring='accuracy')
#     </pre>
#     Does it recover the unbiased score?
#   </li>
# </ul>
# </div>

# %%
# Your code here

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
n_features = 10_000  # many more features than samples

# Pure noise: neither X nor y contain any real signal
X_noise = rng.randn(n_samples, n_features)
y_noise = rng.randint(0, 2, n_samples)  # random binary labels

print(f"X shape: {X_noise.shape}")
print(f"Class balance: {y_noise.mean() * 100:.0f}% positive")

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
print(
    f"Wrong (global feature selection) — accuracy: {scores_fs_wrong.mean():.4f}  "
    f"± {scores_fs_wrong.std():.4f}"
)
print(">>> This is PURE NOISE data — any accuracy above 0.50 is an artefact!")

# %% [markdown]
# #### Baseline expectation
#
# This dataset has **no true signal** — any accuracy significantly above 50%
# is due to leakage, not learning.  Without leakage, we expect ~50% accuracy
# (random chance for binary classification).  Keep this baseline in mind when
# interpreting the scores below: a correct pipeline should hover around chance,
# while a leaky pipeline can climb well above it.

# %% [markdown]
# ### 3.3 The correct way — feature selection inside the pipeline

# %%
# ── CORRECT: selector is re-fitted on each training fold ──────────────────
pipe_fs_correct = make_pipeline(
    SelectKBest(f_classif, k=50), LogisticRegression(max_iter=1000, random_state=42)
)

scores_fs_correct = cross_val_score(
    pipe_fs_correct, X_noise, y_noise, cv=cv_noise, scoring="accuracy"
)
print(
    f"Correct (pipeline) — accuracy: {scores_fs_correct.mean():.4f}  "
    f"± {scores_fs_correct.std():.4f}"
)
print(">>> Now we correctly recover chance-level performance.")

# %% [markdown]
# ### 3.4 Visualising the bias

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(
    ["Wrong\n(global selection)", "Correct\n(pipeline)"],
    [scores_fs_wrong.mean(), scores_fs_correct.mean()],
    yerr=[scores_fs_wrong.std(), scores_fs_correct.std()],
    color=["#d62728", "#2ca02c"],
    capsize=8,
    width=0.4,
)
ax.axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance level (50%)")
ax.set_ylim(0, 1.05)
ax.set_ylabel("CV accuracy")
ax.set_title("Leakage from feature selection on pure-noise data")
ax.legend()
plt.tight_layout()
plt.show()
print(
    f"\nBias introduced by leakage: "
    f"{(scores_fs_wrong.mean() - scores_fs_correct.mean()) * 100:.1f} percentage points"
)

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
    n_samples=200, n_features=100, n_informative=20, noise=20, random_state=42
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
lag_features = np.stack([signal[i : n_time - 5 + i] for i in range(5)], axis=1)
target_ts = signal[5:]  # predict one step ahead

print(f"Time-series features: {lag_features.shape}, target: {target_ts.shape}")

# %% [markdown]
# ### 5.2 Standard KFold vs TimeSeriesSplit

# %%
from sklearn.linear_model import LinearRegression

model_ts = make_pipeline(StandardScaler(), LinearRegression())

# ── WRONG: random folds ignore temporal order ──────────────────────────────
cv_random = KFold(n_splits=5, shuffle=True, random_state=42)
scores_random = cross_val_score(model_ts, lag_features, target_ts, cv=cv_random, scoring="r2")

# ── CORRECT: test folds are always in the future relative to training folds ─
cv_ts = TimeSeriesSplit(n_splits=5)
scores_ts = cross_val_score(model_ts, lag_features, target_ts, cv=cv_ts, scoring="r2")

print(f"Random KFold  R²: {scores_random.mean():.4f} ± {scores_random.std():.4f}")
print(f"TimeSeriesSplit R²: {scores_ts.mean():.4f} ± {scores_ts.std():.4f}")

# %% [markdown]
# ### 5.3 Visualising the split structure

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

for ax, (cv_obj, title) in zip(
    axes, [(cv_random, "KFold (WRONG for time series)"), (cv_ts, "TimeSeriesSplit (CORRECT)")]
):
    for fold, (train_idx, test_idx) in enumerate(cv_obj.split(lag_features)):
        ax.scatter(train_idx, [fold] * len(train_idx), marker="|", color="#2ca02c", alpha=0.4, s=10)
        ax.scatter(test_idx, [fold] * len(test_idx), marker="|", color="#d62728", alpha=0.8, s=10)
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
# ## 6. Diagnosing leakage after the fact
#
# Sometimes you inherit a model or a notebook and suspect leakage but cannot
# re-run the original experiment.  The following signs are strong indicators
# that leakage has inflated the reported performance:
#
# 1. **Performance seems too good to be true.**  If a simple model beats a
#    well-established benchmark by a large margin on a hard task, suspect
#    leakage before celebrating.
# 2. **CV variance is much lower than expected.**  Leakage often stabilises
#    scores across folds because the leaky preprocessing injects the same
#    information into every fold.
# 3. **Test score >> train score.**  A model that generalises *better* than it
#    fits is a classic red flag — it usually means the test set was seen
#    during training-time preprocessing.
# 4. **Feature importance concentrates on a single "leaky" feature.**  If one
#    feature dominates and it was derived from the target (e.g. an encoding,
#    an ID, or a future-looking statistic), that feature is the leak channel.
#
# If you observe these signs, audit your pipeline for any preprocessing,
# encoding, or feature selection performed outside the CV loop, and move it
# inside a `Pipeline`.

# %% [markdown]
# ---
# ## 7. Leakage checklist
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
# **Hint 1.** There are 3 transformations happening outside the CV pipeline.

# %% [markdown]
# **Hint 2.** `StandardScaler`, `PCA`, and `SelectKBest` should each be inside
# the `Pipeline`.

# %% [markdown]
# **Hint 3.** The fix is to wrap all preprocessing in a single `Pipeline`
# before `cross_val_score`.

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
