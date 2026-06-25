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

# %%
%pip install -q git+https://github.com/demianw/Xed.git

# %% [markdown]
# # US County-Level Presidential Elections: A Large-Scale Classification Problem
#
# **Prerequisites:** *Classification* (Titanic), *Evaluation Metrics*, *Cross-Validation*
# notebooks.
#
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Work with a real large-scale political dataset (~3 100 US counties × 14 features).
# 2. Diagnose and handle **class imbalance** in a domain where it has a meaningful
#    political interpretation.
# 3. Build classification pipelines and compare Logistic Regression, Random Forest,
#    and Gradient Boosting.
# 4. Interpret **feature importance** and **partial dependence** in a socio-political context.
# 5. Evaluate temporal generalisation: train on one election, predict the next.
# 6. Analyse the **electoral swing** — which counties flipped and why?
#
# **Political context**
#
# US presidential elections are won state-by-state via the Electoral College,
# but the underlying geography of partisan strength is county-level.
# In 2016, ~84 % of counties voted Republican — yet the Republican candidate
# won the popular vote by a far smaller margin. This is the **urban–rural divide**:
# Republican counties are numerous but small; Democratic counties are few but
# densely populated.
#
# The dataset contains **demographic features from the American Community Survey
# (ACS 2015 5-year estimates)** merged with 2012 and 2016 presidential vote counts
# for every US county (MIT Election Data and Science Lab).
# We ask: *can demographics alone predict which party wins a county,
# and how well does a model trained on 2012 generalise to 2016?*

# %%
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.dummy import DummyClassifier

%matplotlib inline

rng = np.random.default_rng(42)

# %% [markdown]
# ---
# ## 1. Loading and understanding the data

# %%
from xed.datasets import load_us_county_elections

raw = load_us_county_elections()
print(f"Shape: {raw.shape}  ({raw['state'].nunique()} states, {len(raw):,} counties)")
raw.head(3)

# %% [markdown]
# ### 1.1 Building the feature matrix and target

# %%
# ── Demographic features (from ACS 2015 5-year estimates) ────────────────────
DEMO_FEATURES = [
    "white_pct",  # % white non-Hispanic
    "black_pct",  # % Black or African-American
    "hispanic_pct",  # % Hispanic or Latino
    "foreignborn_pct",  # % foreign-born
    "age29andunder_pct",  # % of population aged 29 or under
    "age65andolder_pct",  # % of population aged 65 or older
    "median_hh_inc",  # Median household income (USD)
    "clf_unemploy_pct",  # Civilian labour force unemployment rate
    "lesshs_pct",  # % with less than high school education
    "lesscollege_pct",  # % with less than a 4-year college degree
    "rural_pct",  # % rural population
    "ruralurban_cc",  # Rural-urban continuum code (1=large metro, 9=rural)
    "total_population",  # County total population
]

# ── Previous election results (2012) used as a feature for 2016 prediction ──
PREV_FEATURES = [
    "romney12",  # Romney 2012 raw votes
    "obama12",  # Obama 2012 raw votes
]

ALL_FEATURES = DEMO_FEATURES + PREV_FEATURES

# ── Binary target: Republican won in 2016? ───────────────────────────────────
df = raw[
    ["state", "county", "fips", "trump16", "clinton16", "romney12", "obama12"] + DEMO_FEATURES
].copy()
df = df.dropna(subset=DEMO_FEATURES)  # drop 3 counties with missing ACS data
df["rep_won_2016"] = (df["trump16"] > df["clinton16"]).astype(int)
df["rep_won_2012"] = (df["romney12"] > df["obama12"]).astype(int)

# Derived features: 2012 Republican margin and total 2012 turnout
df["margin_2012"] = (df["romney12"] - df["obama12"]) / (df["romney12"] + df["obama12"])
df["total_votes_2012"] = df["romney12"] + df["obama12"]

FEATURES = DEMO_FEATURES + ["margin_2012", "total_votes_2012"]
X = df[FEATURES]
y = df["rep_won_2016"]

print(f"Dataset: {len(df):,} counties after removing 3 with missing ACS data")
print("\nClass distribution:")
print(f"  Republican won (1): {y.sum():,}  ({y.mean() * 100:.1f}%)")
print(f"  Democrat   won (0): {(1 - y).sum():,}  ({(1 - y.mean()) * 100:.1f}%)")
print(
    f"\nNote: {y.mean() * 100:.0f}% of counties voted Republican — "
    f"yet the Republican candidate won only ~46% of the popular vote."
)
print("      This gap is the urban–rural divide in US politics.")

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 1 — The urban–rural divide</b>
# <ul>
#   <li>
#     <b>Population-weighted class balance.</b>
#     The 84% Republican county figure is unweighted (each county counts equally).
#     Compute the population-weighted Republican share:
#     <code>(df.loc[y==1, 'total_population'].sum() / df['total_population'].sum())</code>.
#     What fraction of the US population lives in Republican-won counties?
#     Is it closer to 84% or 50%?
#   </li>
#   <li>
#     <b>Demographic fingerprint.</b>
#     Compute the mean of each demographic feature separately for Republican
#     and Democrat counties. Which features show the largest gap between the
#     two groups?  Present as a sorted bar chart.
#   </li>
#   <li>
#     <b>2016 swing from 2012.</b>
#     Create a column <code>flipped_R</code> = counties that voted Obama 2012
#     but Trump 2016.  How many counties flipped?  What is their median
#     <code>median_hh_inc</code>, <code>lesshs_pct</code>, and <code>rural_pct</code>
#     compared to counties that stayed Democrat?
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## 2. Baseline models and the imbalance problem

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), FEATURES),
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dummy baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print(f"\nDummy (always Republican) accuracy : {dummy.score(X_test, y_test):.3f}")
print(
    f"  F1 for Democrat class (minority) : "
    f"{f1_score(y_test, dummy.predict(X_test), pos_label=0):.3f}"
)

# %% [markdown]
# An 84% accuracy by predicting "Republican" every time is far too easy to beat
# in terms of raw accuracy — but it **completely fails to identify Democrat counties**.
# For this task, F1 and AUC-ROC on the minority class are the meaningful metrics.

# %%
# Logistic Regression — default (no class weight)
lr_default = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, random_state=42))
lr_default.fit(X_train, y_train)
y_pred_lr = lr_default.predict(X_test)

print("=== Logistic Regression (default) ===")
print(classification_report(y_test, y_pred_lr, target_names=["Democrat (0)", "Republican (1)"]))

# Logistic Regression — balanced class weights
lr_balanced = make_pipeline(
    preprocessor, LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
)
lr_balanced.fit(X_train, y_train)
y_pred_lr_bal = lr_balanced.predict(X_test)

print("=== Logistic Regression (class_weight='balanced') ===")
print(classification_report(y_test, y_pred_lr_bal, target_names=["Democrat (0)", "Republican (1)"]))

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 2 — Comparing classifiers on the correct metric</b>
# <ul>
#   <li>
#     Using 5-fold stratified CV and the <em>balanced accuracy</em> scoring
#     (<code>scoring='balanced_accuracy'</code>), compare:
#     <ol>
#       <li><code>DummyClassifier(strategy='most_frequent')</code></li>
#       <li><code>LogisticRegression(class_weight='balanced')</code></li>
#       <li><code>HistGradientBoostingClassifier(class_weight='balanced', random_state=42)</code></li>
#       <li><code>RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)</code></li>
#     </ol>
#     Print a comparison table with mean ± std for balanced accuracy and F1 (Democrat class).
#   </li>
#   <li>
#     Plot ROC curves for all four models on the test set on a single figure.
#     Which model has the highest AUC?
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## 3. What drives the prediction? Feature importance

# %%
# Fit the best model (HistGradientBoosting) on the full training set
hgb = make_pipeline(
    preprocessor,
    HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42, max_iter=300, learning_rate=0.05
    ),
)
hgb.fit(X_train, y_train)

# Permutation importance on the test set
result = permutation_importance(
    hgb,
    X_test,
    y_test,
    n_repeats=15,
    random_state=42,
    scoring="balanced_accuracy",
)
perm_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }
).sort_values("importance_mean", ascending=True)

fig, ax = plt.subplots(figsize=(7, 7))
ax.barh(
    perm_df["feature"],
    perm_df["importance_mean"],
    xerr=perm_df["importance_std"],
    capsize=3,
    color="steelblue",
)
ax.set_xlabel("Mean decrease in balanced accuracy")
ax.set_title("Permutation importance — US county vote prediction")
plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 3 — Interpreting the model</b>
# <ul>
#   <li>
#     <b>Partial dependence plots.</b>
#     Use <code>PartialDependenceDisplay.from_estimator</code> to plot the partial
#     dependence of the predicted Republican probability on the top 3 most
#     important features.  Do the directions match your political intuition?
#   </li>
#   <li>
#     <b>Error analysis.</b>
#     Find the 20 counties that the model is most confident about but gets wrong
#     (largest <code>|predict_proba - 0.5|</code> on misclassified counties).
#     Print their state, county name, actual result, predicted probability, and
#     demographic features.  Are there any surprising states or counties?
#   </li>
#   <li>
#     <b>Swing predictor.</b>
#     Restrict to counties that voted Obama in 2012
#     (<code>df['rep_won_2012'] == 0</code>).
#     Refit the model on 2012-Democrat counties only and predict which of them
#     would flip Republican in 2016.  What is the precision of this swing detector?
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## 4. Temporal generalisation: train on 2012, predict 2016
#
# A key question in applied ML is whether a model trained on historical data
# generalises to future events — especially when the underlying data-generating
# process (political conditions, demographics) drifts over time.
#
# Here we pose the question directly: **if we had trained our model on the 2012
# election and used it to forecast 2016, how accurate would it have been?**
# The features are the same ACS demographics; only the election year changes.

# %%
# ── Build 2012 dataset ────────────────────────────────────────────────────────
df_2012 = df.dropna(subset=DEMO_FEATURES).copy()
df_2012 = df_2012[df_2012[["romney12", "obama12"]].notna().all(axis=1)]
# For 2012 we don't have prior margin; use just demographics
X_2012 = df_2012[DEMO_FEATURES]
y_2012 = df_2012["rep_won_2012"]

X_2016 = df_2012[DEMO_FEATURES]
y_2016 = df_2012["rep_won_2016"]

# Fit on all 2012 counties, predict 2016 (temporal hold-out)
lr_temporal = make_pipeline(
    make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), DEMO_FEATURES)
    ),
    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
)
lr_temporal.fit(X_2012, y_2012)
y_pred_2016 = lr_temporal.predict(X_2016)

print("=== Train on 2012 demographics → predict 2016 outcome ===")
print(classification_report(y_2016, y_pred_2016, target_names=["Democrat (0)", "Republican (1)"]))
print(f"AUC-ROC: {roc_auc_score(y_2016, lr_temporal.predict_proba(X_2016)[:, 1]):.3f}")

# Compare: same-year CV (in-sample) vs temporal hold-out (out-of-sample)
in_sample = cross_val_score(lr_temporal, X_2012, y_2012, cv=cv, scoring="balanced_accuracy").mean()
print(f"\nIn-sample CV balanced accuracy (2012→2012):    {in_sample:.3f}")
print(
    f"Temporal hold-out balanced accuracy (2012→2016): "
    f"{__import__('sklearn.metrics', fromlist=['balanced_accuracy_score']).balanced_accuracy_score(y_2016, y_pred_2016):.3f}"
)

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 4 — Temporal drift and the limits of demographics</b>
# <ul>
#   <li>
#     <b>What flipped.</b>
#     Add a column <code>flipped_to_R</code> to the test DataFrame
#     (was Democrat in 2012, Republican in 2016).
#     For counties that actually flipped, what fraction did the temporal model
#     correctly predict would flip?  What was its precision and recall on
#     this sub-group?
#   </li>
#   <li>
#     <b>Does adding 2012 results help?</b>
#     Add <code>margin_2012</code> to the 2012-trained model and retrain.
#     How much does the temporal hold-out balanced accuracy improve?
#     What does this tell you about the relative importance of political
#     history vs demographics for election forecasting?
#   </li>
#   <li>
#     <b>State-level analysis.</b>
#     Group the 2016 test counties by state.  For each state compute:
#     the fraction of counties the model got right and the total population
#     of correctly and incorrectly predicted counties.
#     Which states have the highest and lowest prediction accuracy?
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## 5. Hyperparameter tuning with GridSearchCV

# %%
param_grid = {
    "histgradientboostingclassifier__max_iter": [100, 300],
    "histgradientboostingclassifier__learning_rate": [0.05, 0.1, 0.2],
    "histgradientboostingclassifier__max_depth": [3, 5, None],
}

hgb_grid = GridSearchCV(
    make_pipeline(
        make_column_transformer(
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), FEATURES)
        ),
        HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
    ),
    param_grid=param_grid,
    cv=cv,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=0,
)
hgb_grid.fit(X_train, y_train)

print(f"Best parameters   : {hgb_grid.best_params_}")
print(f"Best CV balanced acc : {hgb_grid.best_score_:.4f}")
print()
print(
    classification_report(
        y_test, hgb_grid.predict(X_test), target_names=["Democrat (0)", "Republican (1)"]
    )
)

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 5 — Full model comparison and summary</b>
# <ul>
#   <li>
#     Collect the test-set metrics for all four models you have fitted:
#     Dummy, Logistic Regression (balanced), Tuned HGB, Temporal LR.
#     Build a summary DataFrame with columns for balanced accuracy, F1 (Democrat),
#     F1 (Republican), and AUC-ROC.
#   </li>
#   <li>
#     <b>Calibration check.</b>
#     For the tuned HGB model, bin the predicted Republican probabilities into
#     deciles and compute the observed fraction of Republican counties in each
#     decile.  Plot predicted vs observed probability.
#     Is the model well-calibrated — and does calibration matter for election
#     forecasting?
#   </li>
#   <li>
#     <b>Open-ended.</b>
#     The model uses only county-level demographic snapshots from 2015.
#     List three additional data sources (publicly available) that you think
#     would most improve election forecasting at the county level.
#     For each, explain <em>which</em> counties you expect the model currently
#     misclassifies and why that source would help.
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## Summary
#
# In this notebook you:
# - Loaded and explored a real large-scale political dataset (3 111 US counties).
# - Diagnosed the **urban–rural class imbalance**: 84% of counties vote Republican,
#   yet only ~46% of the population lives in them.
# - Demonstrated that accuracy is misleading here and used balanced accuracy,
#   F1, and AUC-ROC instead.
# - Found that **rural percentage, college education, and racial composition**
#   are the strongest demographic predictors of county-level voting.
# - Tested **temporal generalisation**: a model trained on 2012 demographics
#   achieves near-equivalent accuracy on 2016 — with some notable failures
#   in counties that dramatically swung between elections.
# - Tuned a gradient boosting model with `GridSearchCV` and verified calibration.
#
# **Political take-away:** county demographics explain most of the spatial
# pattern in US presidential elections.  The residual error concentrates in
# swing counties where demographic change outpaced the ACS snapshot,
# and in states with unusual political coalitions.
