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
import os, sys, subprocess

# Install the course package and all pinned dependencies.
# In GitHub Actions CI this step is skipped (pre-installed via pip install -e .[dev]).
if not os.environ.get('CI'):
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q',
         'git+https://github.com/demianw/Xed.git'],
        check=True,
    )
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q',
         'scipy>=1.13',
         ],
        check=True,
    )

# %% [markdown]
# # Motor Insurance Claims: From Cross-Sectional Modelling to Time-Series Prediction
#
# **Prerequisites:** *Classification* (Titanic), *Evaluation Metrics*, *Data Leakage*,
# and *Cross-Validation* notebooks.
#
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Load and explore a real actuarial dataset (freMTPL2 — French motor third-party liability).
# 2. Fit a **Poisson regression** pipeline for claim frequency — the industry-standard GLM.
# 3. Fit a **Tweedie regression** for joint frequency × severity (pure premium).
# 4. Explain the **Bonus-Malus system** as a lagged signal encoding past claim history.
# 5. Simulate a longitudinal policyholder portfolio and build a panel dataset.
# 6. Engineer **lag features** and **rolling statistics** within each policyholder's history.
# 7. Apply **TimeSeriesSplit** instead of random KFold and quantify the leakage from using the wrong splitter.
# 8. Build a full pipeline that predicts next-month claim probability and expected loss per policyholder.
#
# **Insurance context**
#
# Motor insurance pricing rests on two questions asked of every policyholder at every renewal:
# - **Frequency** — how many claims will this driver make per year? (Poisson count model)
# - **Severity** — given a claim, how large will it be? (Log-normal / Gamma model)
#
# Their product is the **pure premium** — the actuarially fair price before expenses and profit margin.
# Modern insurers complement the classical GLM with ML models that exploit the *history*
# of each policyholder: the same driver renewing for the fifth time provides far more
# signal than a new customer.  That historical signal is exactly what time-series modelling captures.

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     TimeSeriesSplit, KFold)
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                             mean_poisson_deviance, d2_tweedie_score,
                             roc_auc_score, f1_score, classification_report)
from sklearn.dummy import DummyClassifier, DummyRegressor

rng = np.random.default_rng(42)

# %% [markdown]
# ---
# ## 1. The freMTPL2 dataset
#
# **freMTPL2** (*French Motor Third-Party Liability, version 2*) is one of the canonical
# benchmarks in actuarial data science.  It contains one record per insurance policy,
# observed over a variable **exposure** period (fraction of a year).
#
# | Column | Description |
# |--------|-------------|
# | `ClaimNb` | Number of claims filed during the exposure period |
# | `Exposure` | Fraction of a year the policy was active (0 – 1) |
# | `Area` | Urban density band (A = rural … F = dense urban) |
# | `VehPower` | Vehicle engine power (ordinal, higher = more powerful) |
# | `VehAge` | Vehicle age in years |
# | `DrivAge` | Driver age in years |
# | `BonusMalus` | Accumulated no-claims discount coefficient (50–350; 100 = neutral) |
# | `VehBrand` | Vehicle brand (anonymised) |
# | `VehGas` | Fuel type (Diesel / Regular) |
# | `Density` | Population density of the policyholder's municipality (persons/km²) |
# | `Region` | French administrative region (anonymised) |
#
# **The Bonus-Malus coefficient** is the most important temporal feature in motor insurance.
# It starts at 100 for a new driver and evolves each renewal year:
# - **−5 %** for each claim-free year (minimum 50 — the "super-bonus")
# - **+25 %** for each year in which at least one claim was filed
#
# A BonusMalus of 50 signals an excellent long-term driving record;
# 230 signals a driver who has repeatedly filed claims and is paying more than double the base premium.

# %%
# Loading 678 013 rows — may take 30–60 s on first call (cached after that)
print("Loading freMTPL2freq …")
ds_freq = fetch_openml(data_id=41214, as_frame=True, parser='auto')
df_freq = ds_freq.frame

print("Loading freMTPL2sev …")
ds_sev  = fetch_openml(data_id=41215, as_frame=True, parser='auto')
df_sev  = ds_sev.frame

# In CI, subsample to keep execution time reasonable
if os.environ.get('CI'):
    df_freq = df_freq.sample(n=50_000, random_state=42).reset_index(drop=True)

print(f"\nFrequency dataset : {df_freq.shape[0]:,} rows × {df_freq.shape[1]} columns")
print(f"Severity dataset  : {df_sev.shape[0]:,} rows × {df_sev.shape[1]} columns")
df_freq.head(3)

# %% [markdown]
# ### 1.1 Key actuarial statistics

# %%
total_exposure = df_freq['Exposure'].sum()
total_claims   = df_freq['ClaimNb'].sum()
freq_per_year  = total_claims / total_exposure

print(f"Total policy-years of exposure : {total_exposure:,.0f}")
print(f"Total claims                   : {total_claims:,.0f}")
print(f"Observed frequency (per year)  : {freq_per_year:.4f}  "
      f"≈ one claim every {1/freq_per_year:.1f} years")
print(f"Policies with 0 claims         : {(df_freq['ClaimNb'] == 0).mean()*100:.1f}%")
print(f"Policies with ≥ 1 claim        : {(df_freq['ClaimNb'] >= 1).mean()*100:.1f}%")
print()
print("BonusMalus distribution:")
print(df_freq['BonusMalus'].describe().to_string())

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 1 — Exploratory analysis</b>
# <ul>
#   <li>
#     <b>Claim frequency by driver age.</b> Bin <code>DrivAge</code> into the groups
#     18–24, 25–34, 35–49, 50–64, 65–79, 80+ using <code>pd.cut</code>.
#     For each group compute the <em>exposure-weighted</em> claim frequency:
#     <code>ClaimNb.sum() / Exposure.sum()</code>.
#     Plot as a bar chart.  Which age groups are riskiest?  Does this match
#     your intuition about motor insurance pricing?
#   </li>
#   <li>
#     <b>BonusMalus as a risk signal.</b> Bin <code>BonusMalus</code> into deciles.
#     For each decile compute the exposure-weighted claim frequency.
#     Is BonusMalus a monotone predictor of risk?  Plot frequency vs BonusMalus decile.
#   </li>
#   <li>
#     <b>Urban vs rural risk.</b> The <code>Area</code> column runs from A (rural) to F
#     (dense urban).  Compute claim frequency and mean severity
#     (joining with <code>df_sev</code> on <code>IDpol</code>) for each area band.
#     Is the urban risk premium mainly driven by frequency, severity, or both?
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## 2. Claim frequency modelling — Poisson regression
#
# The actuarial standard model for claim counts is the **Poisson GLM with log link**:
#
# $$\mathbb{E}[\text{ClaimNb}_i] = \text{Exposure}_i \cdot \exp(\mathbf{x}_i^\top \boldsymbol{\beta})$$
#
# The **exposure offset** is key: a policy observed for only 6 months (Exposure = 0.5)
# should be expected to produce half as many claims as one observed for a full year,
# *before* any feature effects.  In scikit-learn this is handled by dividing the target
# by exposure and passing `sample_weight=exposure` to the regressor — equivalent to
# fitting on the **frequency** $y_i = \text{ClaimNb}_i / \text{Exposure}_i$.
#
# The `PoissonRegressor` minimises the **Poisson deviance** (not MSE), which is the
# correct loss for count data with many zeros and a skewed distribution.

# %%
# Prepare frequency target and sample weights
y_freq = df_freq['ClaimNb'] / df_freq['Exposure']   # annualised claim frequency
w_freq = df_freq['Exposure']                          # sample weights

# Clean up VehGas: strip stray quotes introduced by OpenML
df_freq['VehGas'] = df_freq['VehGas'].str.strip("'")

# Feature matrix
X = df_freq.drop(columns=['IDpol', 'ClaimNb', 'Exposure'])

# Train / test split — 80 / 20, stratified on zero/non-zero claims
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y_freq, w_freq, test_size=0.20, random_state=42
)

print(f"Train : {len(X_train):,} policies")
print(f"Test  : {len(X_test):,} policies")

# %% [markdown]
# ### 2.1 Building the preprocessing pipeline

# %%
numeric_cols     = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
categorical_cols = ['Area', 'VehBrand', 'VehGas', 'Region']

numeric_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
categorical_pipe = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
)

preprocessor = make_column_transformer(
    (numeric_pipe,     numeric_cols),
    (categorical_pipe, categorical_cols),
)

# Poisson regression pipeline
poisson_pipe = make_pipeline(
    preprocessor,
    PoissonRegressor(alpha=1e-3, max_iter=500),
)

poisson_pipe.fit(X_train, y_train, poissonregressor__sample_weight=w_train)

# %% [markdown]
# ### 2.2 Evaluating the frequency model
#
# The standard evaluation metric for Poisson frequency models is the
# **Poisson deviance** (a generalisation of MSE for count data), which sklearn
# exposes as `mean_poisson_deviance`.  The **D² score** (analogous to R²) measures
# the fraction of deviance explained relative to the null model (intercept only).

# %%
y_pred_train = poisson_pipe.predict(X_train)
y_pred_test  = poisson_pipe.predict(X_test)

dev_train = mean_poisson_deviance(y_train, y_pred_train, sample_weight=w_train)
dev_test  = mean_poisson_deviance(y_test,  y_pred_test,  sample_weight=w_test)
d2_train  = d2_tweedie_score(y_train, y_pred_train, sample_weight=w_train, power=1)
d2_test   = d2_tweedie_score(y_test,  y_pred_test,  sample_weight=w_test,  power=1)

print("Poisson GLM — frequency model")
print(f"  Poisson deviance  : train = {dev_train:.4f}   test = {dev_test:.4f}")
print(f"  D² score          : train = {d2_train:.4f}   test = {d2_test:.4f}")
print()
# Calibration check: predicted mean should match observed mean
print(f"  Mean predicted freq : {(y_pred_test * w_test).sum() / w_test.sum():.5f}")
print(f"  Observed freq       : {(y_test      * w_test).sum() / w_test.sum():.5f}")

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 2 — Poisson regression and model comparison</b>
# <ul>
#   <li>
#     <b>Inspect the BonusMalus coefficient.</b>
#     Extract the fitted <code>PoissonRegressor</code> from the pipeline with
#     <code>poisson_pipe[-1]</code>.  The <code>coef_</code> array corresponds to
#     the preprocessed feature order: first the <em>numeric_cols</em> (scaled), then
#     the one-hot columns.  Print the coefficient for BonusMalus (first numeric column
#     after the scaler — index 0 before one-hot expansion).
#     Is the sign what you would expect for a risk factor?
#   </li>
#   <li>
#     <b>Compare against a gradient boosted model.</b>
#     <code>HistGradientBoostingRegressor</code> handles missing values natively
#     and supports <code>sample_weight</code>.  Build a pipeline with it and compare
#     Poisson deviance and D² on the test set.  Which model fits better?
#     <br>Hint: use <code>loss='poisson'</code> so the loss is also Poisson deviance.
#   </li>
#   <li>
#     <b>Lift curve.</b> Sort the test set by descending predicted frequency.
#     Compute the cumulative observed claims (weighted by exposure) for the top 10 %,
#     20 %, … 100 % of policies.  A good model concentrates observed claims in the
#     top-ranked policies.  Plot this "lift curve" for both the Poisson GLM and the
#     gradient boosted model.
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## 3. Claim severity and the pure premium
#
# The **pure premium** (also called "risk premium") is what the insurer must collect
# on average to break even on claims:
#
# $$\text{Pure premium} = \underbrace{\lambda}_{\text{frequency}} \times
#   \underbrace{\mu}_{\text{mean severity}}$$
#
# We model severity using a **Tweedie regression** with `power` between 1 and 2.
# This family nests:
# - `power=1` → Poisson (for counts/rates)
# - `power=1.5` → compound Poisson-Gamma (the most common choice for pure premiums)
# - `power=2` → Gamma (for strictly positive outcomes)
#
# The Tweedie model with `power=1.5` directly models the **pure premium**
# (zero-inflated continuous variable) in a single step, avoiding the separate
# frequency/severity decomposition.

# %%
# Merge severity onto frequency data
df_merged = df_freq.merge(
    df_sev.groupby('IDpol')['ClaimAmount'].sum().reset_index(),
    on='IDpol', how='left'
)
df_merged['ClaimAmount'] = df_merged['ClaimAmount'].fillna(0.0)

# Pure premium target: total claim amount per unit of exposure
y_pp = df_merged['ClaimAmount'] / df_merged['Exposure']

X_pp = df_merged.drop(columns=['IDpol', 'ClaimNb', 'Exposure', 'ClaimAmount'])
w_pp = df_merged['Exposure']

X_pp_train, X_pp_test, y_pp_train, y_pp_test, w_pp_train, w_pp_test = train_test_split(
    X_pp, y_pp, w_pp, test_size=0.20, random_state=42
)

# Tweedie (compound Poisson-Gamma) pipeline
tweedie_pipe = make_pipeline(
    preprocessor,
    TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000, link='log'),
)
tweedie_pipe.fit(X_pp_train, y_pp_train, tweedieregressor__sample_weight=w_pp_train)

d2_tweedie = d2_tweedie_score(
    y_pp_test, tweedie_pipe.predict(X_pp_test),
    sample_weight=w_pp_test, power=1.5
)
print(f"Tweedie GLM (power=1.5) D² score : {d2_tweedie:.4f}")
print(f"Mean predicted pure premium : €{tweedie_pipe.predict(X_pp_test).mean():.2f} / year")
print(f"Mean observed  pure premium : €{(y_pp_test * w_pp_test).sum() / w_pp_test.sum():.2f} / year")

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 3 — Pure premium model</b>
# <ul>
#   <li>
#     Compare the Tweedie GLM with a
#     <code>HistGradientBoostingRegressor(loss='gamma')</code>
#     on the subset of policies with at least one claim
#     (filtering <code>y_pp > 0</code>).  Report MAE, RMSE, and D² (power=2 for Gamma).
#     Does the non-parametric model capture severity better?
#   </li>
#   <li>
#     <b>Decile analysis.</b> Bin the test set into predicted-pure-premium deciles.
#     For each decile, compute the mean predicted pure premium and the mean
#     observed pure premium.  A perfectly calibrated model should lie on the
#     diagonal.  Plot predicted vs observed mean per decile.
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## 4. The Bonus-Malus system as a time-series signal
#
# The most important feature in the freMTPL2 dataset is `BonusMalus` — but where
# does it come from?  It is the output of a **recursive update rule** that encodes
# the policyholder's entire past claim history into a single number:
#
# $$\text{BM}_{t+1} =
# \begin{cases}
#   \max(50,\ \text{BM}_t \times 0.95) & \text{if no claim in year } t \\
#   \min(350,\ \text{BM}_t \times 1.25^{k}) & \text{if } k \ge 1 \text{ claims in year } t
# \end{cases}$$
#
# In statistical terms, BonusMalus is a **sufficient statistic** of past claim history
# under the Poisson-Gamma credibility model — it is exactly what a Bayesian actuary
# would compute as the posterior mean risk estimate given all past observations.
#
# This recursive structure means that if we observe a policyholder across multiple
# renewal years, BonusMalus *is* a lag feature: it summarises everything that
# happened before the current observation.  This makes it the natural bridge between
# cross-sectional GLMs and longitudinal machine-learning models.

# %%
# Illustrate BonusMalus dynamics for three stylised drivers

def update_bm(bm, n_claims):
    """Apply French BM update rule for one year."""
    if n_claims == 0:
        return max(50.0, bm * 0.95)
    else:
        return min(350.0, bm * (1.25 ** n_claims))

n_years = 15
profiles = {
    'Safe driver (0 claims/yr)':   [0] * n_years,
    'Average driver (~1 claim/3yr)': [0,0,1,0,0,1,0,0,0,1,0,0,0,0,1],
    'Risky driver (~1 claim/yr)':  [1,0,1,1,0,1,1,0,1,1,1,0,1,1,0],
}

fig, ax = plt.subplots(figsize=(10, 4))
for label, claim_history in profiles.items():
    bm, trajectory = 100.0, [100.0]
    for claims in claim_history:
        bm = update_bm(bm, claims)
        trajectory.append(bm)
    ax.plot(range(n_years + 1), trajectory, marker='o', markersize=4, label=label)

ax.axhline(100, color='grey', linestyle='--', linewidth=0.8, label='Neutral (100)')
ax.axhline(50,  color='green', linestyle=':',  linewidth=0.8, label='Super-bonus (50)')
ax.set_xlabel('Year')
ax.set_ylabel('BonusMalus coefficient')
ax.set_title('BonusMalus trajectory for three driver profiles (15 years)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 4 — BonusMalus as a predictor</b>
# <ul>
#   <li>
#     A new policyholder always starts at BonusMalus = 100.
#     After how many consecutive claim-free years does a driver first reach
#     the super-bonus of 50?  Write a short loop using <code>update_bm</code>
#     and print the year-by-year trajectory.
#   </li>
#   <li>
#     A driver at BonusMalus = 50 files two claims in one year.
#     Simulate their BonusMalus over the next 10 years assuming they then
#     file zero claims.  How many years does it take to return to 50?
#     What does this tell you about the "memory" built into the system?
#   </li>
#   <li>
#     In the freMTPL2freq dataset, sort policies by <code>BonusMalus</code>
#     and compute mean observed claim frequency for each integer BM value
#     (or for BM bins of width 10).  Is the relationship linear on a log scale?
#     Plot log-frequency vs BonusMalus and add a linear regression line.
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## 5. Portfolio simulation — building a longitudinal dataset
#
# The freMTPL2 dataset is **cross-sectional**: one row per policy, one year of exposure.
# To study time-series methods, we need to observe the *same* policyholders across
# multiple consecutive years so their claim history and BonusMalus evolve over time.
#
# We build this longitudinal dataset by:
# 1. Sampling a cohort of 600 policyholders from freMTPL2freq.
# 2. Using the fitted Poisson GLM to assign each policyholder a latent annual
#    claim rate $\hat{\lambda}_i$.
# 3. Simulating 36 monthly observations per policyholder, updating BonusMalus
#    annually using the French update rule.
#
# The resulting panel dataset has genuine temporal structure: claim events in past
# months causally influence future BonusMalus and therefore future claim rates.

# %%
def simulate_portfolio(
    poisson_pipe: object,
    source_df: pd.DataFrame,
    n_policyholders: int = 600,
    n_months: int = 36,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a longitudinal panel from a fitted Poisson frequency pipeline.

    Parameters
    ----------
    poisson_pipe : fitted sklearn Pipeline
        Must expose a ``predict`` method returning annualised claim frequency.
    source_df : pd.DataFrame
        The freMTPL2freq frame to sample policyholder profiles from.
    n_policyholders : int
        Number of policyholders to track.
    n_months : int
        Number of consecutive monthly observations per policyholder.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel dataset sorted by (policyholder_id, month_index), with columns:
        policyholder_id, month_index, calendar_year, calendar_month,
        BonusMalus, lambda_annual, p_claim, claim_occurred, claim_amount,
        plus all static policyholder features.
    """
    rng_sim = np.random.default_rng(seed)

    # ── 1. Sample a cohort of policyholders ────────────────────────────
    feature_cols = [c for c in source_df.columns
                    if c not in ('IDpol', 'ClaimNb', 'Exposure')]
    cohort = (source_df[feature_cols]
              .sample(n=n_policyholders, random_state=seed, replace=False)
              .reset_index(drop=True))
    cohort['VehGas'] = cohort['VehGas'].str.strip("'")

    # ── 2. Assign initial BonusMalus from the sampled profiles ─────────
    bm = cohort['BonusMalus'].values.astype(float)

    # ── 3. Static features (everything except BonusMalus and Exposure) ─
    static_cols = [c for c in feature_cols if c not in ('BonusMalus',)]
    static = cohort[static_cols].copy()

    # Seasonality multipliers (higher claims in winter months)
    season = {1: 1.30, 2: 1.25, 3: 1.05, 4: 0.90, 5: 0.85, 6: 0.80,
              7: 0.80, 8: 0.85, 9: 0.95, 10: 1.05, 11: 1.20, 12: 1.35}

    records = []
    year_claim_counts = np.zeros(n_policyholders, dtype=int)  # within current year

    for month_idx in range(n_months):
        cal_month = month_idx % 12 + 1   # 1–12
        cal_year  = 2020 + month_idx // 12

        # Build the feature frame for this month (update BonusMalus column)
        X_month = static.copy()
        X_month['BonusMalus'] = bm

        # Predict annualised claim frequency from the GLM
        lambda_annual = poisson_pipe.predict(X_month)

        # Monthly claim probability with seasonal adjustment
        s = season[cal_month]
        p_claim = 1.0 - np.exp(-lambda_annual * s / 12.0)
        p_claim = np.clip(p_claim, 0.0, 0.99)

        # Simulate claim occurrence
        claim_occ = rng_sim.binomial(1, p_claim).astype(int)

        # Simulate claim amount for policies with a claim
        # Log-normal severity: median ≈ €1 200, with vehicle-age loading
        veh_age = static['VehAge'].values
        sigma_sev = 0.75
        mu_sev = np.log(1200) + 0.03 * veh_age   # older vehicles → higher repair cost
        raw_sev = rng_sim.lognormal(mu_sev, sigma_sev)
        claim_amount = claim_occ * raw_sev

        # Store one row per policyholder for this month
        for i in range(n_policyholders):
            row = {
                'policyholder_id': i,
                'month_index':     month_idx,
                'calendar_year':   cal_year,
                'calendar_month':  cal_month,
                'BonusMalus':      bm[i],
                'lambda_annual':   lambda_annual[i],
                'p_claim':         p_claim[i],
                'claim_occurred':  claim_occ[i],
                'claim_amount':    claim_amount[i],
            }
            for col in static_cols:
                row[col] = static[col].iloc[i]
            records.append(row)

        # Accumulate annual claim counts
        year_claim_counts += claim_occ

        # Update BonusMalus at the end of each calendar year
        if cal_month == 12:
            bm = np.where(
                year_claim_counts == 0,
                np.maximum(50.0,  bm * 0.95),
                np.minimum(350.0, bm * (1.25 ** year_claim_counts)),
            )
            year_claim_counts[:] = 0   # reset counter for the new year

    panel = pd.DataFrame(records)
    panel = panel.sort_values(['policyholder_id', 'month_index']).reset_index(drop=True)
    return panel

print("Simulating 600 policyholders × 36 months …")
panel = simulate_portfolio(poisson_pipe, df_freq, n_policyholders=600, n_months=36)
print(f"Panel shape : {panel.shape}")
print(f"Total claims in simulation : {panel['claim_occurred'].sum()}")
print(f"Overall monthly claim rate : {panel['claim_occurred'].mean()*100:.2f}%")
panel.head(6)

# %% [markdown]
# ---
# ## 6. Feature engineering for time series
#
# Unlike cross-sectional models, longitudinal models can exploit
# **what happened to this policyholder in previous months**.
# We add three kinds of temporal features:
#
# | Feature | Description | Why it matters |
# |---------|-------------|----------------|
# | `prev_claim` | Did the policyholder claim last month? | Immediate recency signal |
# | `claims_3m`  | Number of claims in the past 3 months | Short-term frequency trend |
# | `claims_12m` | Number of claims in the past 12 months | Input to BM update rule |
# | `month_sin`, `month_cos` | Cyclical encoding of calendar month | Captures seasonal pattern without a spurious discontinuity between December and January |
#
# **Critical rule:** lag features must be computed *within each policyholder* and
# sorted by time.  Computing a rolling mean across policyholder boundaries would
# introduce information from a different person's claim history — pure leakage.

# %%
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and cyclical features to a panel dataset.

    The panel must be sorted by (policyholder_id, month_index) before calling.
    All shift/rolling operations are performed within each policyholder group
    to prevent information from bleeding across individuals.
    """
    df = df.copy().sort_values(['policyholder_id', 'month_index'])

    grp = df.groupby('policyholder_id')['claim_occurred']

    # Lag features (shift by 1 so there is no same-month leakage)
    df['prev_claim']  = grp.shift(1).fillna(0).astype(int)
    df['claims_3m']   = (grp.shift(1)
                            .rolling(window=3,  min_periods=1)
                            .sum()
                            .reset_index(level=0, drop=True)
                            .fillna(0))
    df['claims_12m']  = (grp.shift(1)
                            .rolling(window=12, min_periods=1)
                            .sum()
                            .reset_index(level=0, drop=True)
                            .fillna(0))

    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['calendar_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['calendar_month'] / 12)

    # Year trend (0, 1, 2 for the three simulated years)
    df['year_idx'] = df['calendar_year'] - df['calendar_year'].min()

    return df

panel_feat = add_temporal_features(panel)
print("New temporal features:")
print(panel_feat[['policyholder_id', 'month_index', 'calendar_month',
                   'prev_claim', 'claims_3m', 'claims_12m',
                   'month_sin', 'month_cos', 'claim_occurred']].head(15).to_string())

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 5 — Feature engineering</b>
# <ul>
#   <li>
#     <b>Validate no cross-policyholder leakage.</b>
#     For policyholder 0, print rows 10–14.
#     For policyholder 1, print rows 0–4.
#     Verify that <code>claims_3m</code> for month 0 of policyholder 1
#     does <em>not</em> reflect the last three months of policyholder 0.
#   </li>
#   <li>
#     <b>Predictive power of lag features.</b>
#     Group the panel by <code>claims_3m</code> (binned into 0, 1, 2, 3+)
#     and compute the mean <code>claim_occurred</code> in each group.
#     Plot as a bar chart.  Does recent claim history predict the current month's
#     claim probability, on top of what BonusMalus already encodes?
#   </li>
#   <li>
#     <b>Seasonality check.</b> Group by <code>calendar_month</code> and compute
#     the mean <code>claim_occurred</code>.  Plot as a bar chart.
#     Does the seasonal pattern in the simulation match what you would expect
#     from real motor insurance data in Western Europe?
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## 7. The temporal split — why KFold leaks on sequential data
#
# The panel dataset is a time series of 36 monthly snapshots.
# Randomly shuffling rows before splitting breaks temporal ordering:
# a fold's training set may contain months **after** its test set.
# This means the model implicitly learns future information during training
# — the same leakage demonstrated in the *Data Leakage* notebook, but in
# the panel domain.
#
# The correct approach is to ensure that every test fold comes
# **strictly after** every training fold in calendar time.
# `TimeSeriesSplit` guarantees this by expanding the training window forward.
#
# For panel data the rows must be **sorted by `month_index`** (not by
# `policyholder_id`) before passing to `TimeSeriesSplit`, so that the
# splitter respects calendar time rather than policyholder identity.

# %%
# ── Prepare the modelling dataset ──────────────────────────────────────────────
# Drop the first 3 months (no lag-3 history) and sort by time, then policyholder
panel_model = (panel_feat[panel_feat['month_index'] >= 3]
               .sort_values(['month_index', 'policyholder_id'])
               .reset_index(drop=True))

feature_cols_ts = [
    'BonusMalus', 'VehPower', 'VehAge', 'DrivAge', 'Density',
    'prev_claim', 'claims_3m', 'claims_12m',
    'month_sin', 'month_cos', 'year_idx',
    'Area', 'VehBrand', 'VehGas', 'Region',
]
X_panel = panel_model[feature_cols_ts]
y_panel = panel_model['claim_occurred']

# ── Compare KFold (wrong) vs TimeSeriesSplit (correct) ────────────────────────
from sklearn.linear_model import LogisticRegression

panel_preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()),
     ['BonusMalus', 'VehPower', 'VehAge', 'DrivAge', 'Density',
      'prev_claim', 'claims_3m', 'claims_12m',
      'month_sin', 'month_cos', 'year_idx']),
    (make_pipeline(SimpleImputer(strategy='most_frequent'),
                   OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
     ['Area', 'VehBrand', 'VehGas', 'Region']),
)

lr_pipe = make_pipeline(
    panel_preprocessor,
    LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
)

# TimeSeriesSplit: 5 folds, each test fold = 5 months, gap = 1 month
# The gap prevents the model from using the month immediately before the test window
# (which may be correlated by the BM update cycle).
tscv = TimeSeriesSplit(n_splits=5, gap=600)   # gap=600 rows ≈ 1 month × 600 policyholders
kfcv = KFold(n_splits=5, shuffle=True, random_state=42)

scores_ts  = cross_val_score(lr_pipe, X_panel, y_panel, cv=tscv, scoring='roc_auc')
scores_kf  = cross_val_score(lr_pipe, X_panel, y_panel, cv=kfcv, scoring='roc_auc')

print("Logistic Regression — AUC-ROC")
print(f"  TimeSeriesSplit : {scores_ts.mean():.4f} ± {scores_ts.std():.4f}")
print(f"  KFold (wrong)   : {scores_kf.mean():.4f} ± {scores_kf.std():.4f}")
print(f"  Leakage bias    : {scores_kf.mean() - scores_ts.mean():+.4f}")

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 6 — Comparing classifiers with the right CV strategy</b>
# <ul>
#   <li>
#     Using <code>TimeSeriesSplit</code> as the CV strategy, compare:
#     <ol>
#       <li><code>DummyClassifier(strategy='most_frequent')</code></li>
#       <li><code>LogisticRegression(class_weight='balanced')</code> (already fitted)</li>
#       <li><code>HistGradientBoostingClassifier(random_state=42)</code></li>
#     </ol>
#     Use both <code>roc_auc</code> and <code>f1</code> as scoring metrics.
#     Print a comparison table.
#   </li>
#   <li>
#     <b>Which features matter most?</b>
#     Fit the best classifier on the first 30 months (training set)
#     and compute permutation importances on the last 6 months (test set).
#     Do the temporal features (<code>claims_3m</code>, <code>BonusMalus</code>)
#     rank above the static features (<code>VehAge</code>, <code>DrivAge</code>)?
#   </li>
#   <li>
#     <b>Calibration.</b> Plot the probability calibration curve (reliability diagram)
#     for the best classifier on the held-out test set.  Is the model well-calibrated?
#     An insurer needs well-calibrated probabilities because they directly enter
#     the premium calculation.
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## 8. Expected loss: combining frequency and severity
#
# In practice, the insurer needs a single monetary estimate per policyholder:
#
# $$\hat{L}_i = \underbrace{\hat{p}_i}_{\text{P(claim)}} \times
#   \underbrace{\hat{\mu}_i}_{\text{E[amount | claim]}}$$
#
# We predict frequency using the classification pipeline from Section 7,
# and severity using a regression model trained on the subset of months
# where a claim actually occurred.

# %%
# ── Train / test split by time ──────────────────────────────────────────────
cutoff = panel_model['month_index'].max() - 5  # last 6 months = test
train_mask = panel_model['month_index'] <  cutoff
test_mask  = panel_model['month_index'] >= cutoff

X_train_ts = X_panel[train_mask]
X_test_ts  = X_panel[test_mask]
y_cls_train = panel_model.loc[train_mask, 'claim_occurred']
y_cls_test  = panel_model.loc[test_mask,  'claim_occurred']
y_sev_train = panel_model.loc[train_mask, 'claim_amount']
y_sev_test  = panel_model.loc[test_mask,  'claim_amount']

# ── Frequency model (classification) ─────────────────────────────────────────
freq_clf = make_pipeline(
    panel_preprocessor,
    HistGradientBoostingClassifier(random_state=42),
)
freq_clf.fit(X_train_ts, y_cls_train)
p_claim_test = freq_clf.predict_proba(X_test_ts)[:, 1]

# ── Severity model (regression on claim months only) ─────────────────────────
claim_train = train_mask & (panel_model['claim_occurred'] == 1)
sev_reg = make_pipeline(
    panel_preprocessor,
    HistGradientBoostingRegressor(loss='gamma', random_state=42),
)
sev_reg.fit(X_panel[claim_train], y_sev_train[claim_train.values])
mu_sev_test = sev_reg.predict(X_test_ts)

# ── Expected loss per policyholder-month ─────────────────────────────────────
panel_test = panel_model[test_mask].copy()
panel_test['p_claim_pred'] = p_claim_test
panel_test['mu_sev_pred']  = mu_sev_test
panel_test['expected_loss'] = panel_test['p_claim_pred'] * panel_test['mu_sev_pred']

# Aggregate over the 6-month test window per policyholder
agg = (panel_test.groupby('policyholder_id')
       .agg(
           total_expected_loss=('expected_loss', 'sum'),
           total_observed_loss=('claim_amount',  'sum'),
           n_claims_observed  =('claim_occurred','sum'),
           mean_bm            =('BonusMalus',    'mean'),
           driv_age           =('DrivAge',        'first'),
       )
       .sort_values('total_expected_loss', ascending=False)
       .reset_index())

print("Top 10 highest-risk policyholders (6-month test window)")
print(agg.head(10).to_string(index=False))

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 7 — Risk ranking and premium loading</b>
# <ul>
#   <li>
#     <b>Model calibration.</b>
#     Divide the test set into deciles by <code>expected_loss</code>.
#     For each decile, compute the mean predicted expected loss and the mean
#     observed claim amount.  Plot predicted vs observed — how well-calibrated
#     is the combined model?
#   </li>
#   <li>
#     <b>Premium loading.</b>
#     Insurers add a <em>safety loading</em> above the pure premium to cover
#     uncertainty and profit margin.  A common approach is:
#     <code>premium = expected_loss × (1 + loading)</code>.
#     <br>
#     Compute <code>loading</code> such that the total premium collected over all
#     test policyholders exactly covers the total observed losses plus a 15 % margin.
#     Is this loading uniform across risk deciles, or should riskier policyholders
#     pay a higher loading to reflect model uncertainty?
#   </li>
#   <li>
#     <b>Static vs dynamic model.</b>
#     Refit the frequency classifier <em>without</em> the temporal features
#     (<code>prev_claim</code>, <code>claims_3m</code>, <code>claims_12m</code>)
#     and recompute the expected-loss rankings.
#     How many of the top-10 riskiest policyholders change?
#     What does this tell you about the value of claim history in pricing?
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
# - Loaded **freMTPL2** — the reference actuarial benchmark for French motor insurance —
#   and computed exposure-weighted claim frequencies by driver age, BonusMalus, and area.
# - Fitted a **Poisson GLM** (the industry-standard frequency model) and evaluated it with
#   Poisson deviance and D² score, including a calibration check.
# - Extended to a **Tweedie regression** (power = 1.5) for the joint frequency × severity
#   (pure premium), avoiding the need to model frequency and severity separately.
# - Explained the **Bonus-Malus update rule** as a recursive time-series signal that
#   summarises a policyholder's entire claim history into a single number.
# - **Simulated a longitudinal portfolio** of 600 policyholders over 36 months,
#   using the fitted GLM as the data-generating process and the French BM rule for
#   temporal dynamics.
# - Engineered **lag features** (`prev_claim`, `claims_3m`, `claims_12m`) and
#   **cyclical month features** (sin/cos encoding), with explicit validation that no
#   information crossed policyholder boundaries.
# - Demonstrated that **KFold leaks future information** on sequential panel data and
#   quantified the resulting AUC bias against `TimeSeriesSplit`.
# - Built a combined **frequency × severity expected-loss model**, ranked policyholders
#   by predicted risk, and computed the premium loading needed to achieve a 15 % margin.
#
# **Key actuarial take-away:** the Bonus-Malus coefficient alone captures most of the
# predictive signal available from a policyholder's history.  Adding explicit lag features
# on top of BonusMalus provides incremental lift, particularly for detecting short-term
# claim bursts that have not yet propagated into the BM coefficient.
