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
# %pip install -q scikit-learn==1.9.0 pandas==3.0.3 matplotlib==3.11.0 seaborn==0.13.2 pooch

# %%
import os, sys
# Skip clone/chdir when running under GitHub Actions (already in the right directory).
if not os.environ.get('CI'):
    if not os.path.exists('Xed'):
        os.system('git clone --depth=1 https://github.com/demianw/Xed.git')
    if 'evaluation_metrics' not in os.getcwd():
        os.chdir('Xed/evaluation_metrics')
# Make `xed.datasets` importable from any subdirectory.
_repo_root = os.path.abspath('..')
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# %% [markdown]
# # Evaluation Metrics: Beyond Accuracy
#
# **Prerequisites:** complete the *Titanic Survival Prediction* and *Ames Housing
# Regression* notebooks first. The datasets and preprocessing patterns from those
# notebooks are reused here.
#
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Explain why accuracy alone is an unreliable performance metric for
#    classification, especially on imbalanced datasets.
# 2. Read and interpret a confusion matrix and derive precision, recall, and F1.
# 3. Plot and compare ROC curves; compute and interpret AUC-ROC.
# 4. Choose the appropriate scoring metric when calling `cross_val_score`.
# 5. Compute and compare regression metrics (RMSE, MAE, R²) and understand
#    their different sensitivities to outliers.
# 6. Diagnose class-imbalance problems and apply corrective strategies.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier

rng = np.random.RandomState(42)

# %% [markdown]
# ---
# ## Part 1 — Classification Metrics

# %% [markdown]
# ## 1. Why accuracy is not enough
#
# We start by loading the Titanic dataset and building a reference preprocessing
# pipeline identical to the one from the classification notebook.

# %%
from xed.datasets import load_titanic, load_ames_housing
titanic = load_titanic()

features = titanic.drop(columns='Survived')
target   = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.20, random_state=42, stratify=target
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print()
print("Class distribution in the test set:")
print(y_test.value_counts(normalize=True).rename({0: 'Did not survive', 1: 'Survived'}).to_string())

# %% [markdown]
# About **38 %** of passengers survived and **62 %** did not.
# That gap matters: a classifier that *always* predicts "did not survive"
# would be correct 62 % of the time without having learned anything at all.
#
# Let us verify this with scikit-learn's `DummyClassifier`:

# %%
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_acc = dummy.score(X_test, y_test)

print(f"Dummy classifier accuracy : {dummy_acc:.3f}")
print("This classifier *never* predicts a survivor.")

# %% [markdown]
# **62 % accuracy for free.** Any real model must beat this baseline
# *and* we need metrics that reveal *how* it beats it.
#
# ### Reference preprocessing pipeline
#
# We reproduce the pipeline from the Titanic notebook so the exercises below
# can focus entirely on metrics rather than feature engineering.

# %%
numeric_features    = ['Age', 'Fare', 'SibSp', 'Parch']
ordinal_features    = ['Pclass']
categorical_features = ['Sex', 'Embarked']

numeric_transformer = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
)
categorical_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore'),
)
ordinal_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OrdinalEncoder(),
)

preprocessor = make_column_transformer(
    (numeric_transformer,    numeric_features),
    (ordinal_transformer,    ordinal_features),
    (categorical_transformer, categorical_features),
)

# Logistic regression baseline
lr_pipeline = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=1000, random_state=42),
)
lr_pipeline.fit(X_train, y_train)
lr_acc = lr_pipeline.score(X_test, y_test)

print(f"Logistic Regression accuracy : {lr_acc:.3f}")
print(f"Improvement over dummy       : {lr_acc - dummy_acc:+.3f}")

# %% [markdown]
# The logistic regression is roughly 18 percentage points better than the dummy.
# But *what kind* of mistakes does it make? Accuracy alone cannot tell us.

# %% [markdown]
# ---
# ## 2. The confusion matrix
#
# The **confusion matrix** counts every combination of true and predicted label.
# For a binary problem it has four cells:
#
# |                      | Predicted: did not survive | Predicted: survived |
# |----------------------|---------------------------|---------------------|
# | **True: did not survive** | True Negative (TN)   | False Positive (FP) |
# | **True: survived**        | False Negative (FN)  | True Positive (TP)  |
#
# - **False Positive (FP)** — the model says "survived" but the passenger actually died.
# - **False Negative (FN)** — the model says "died" but the passenger actually survived.
#
# Whether FP or FN is more costly depends on the application.
# In a medical diagnosis context these costs are very different.

# %%
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay.from_estimator(
    lr_pipeline, X_test, y_test,
    display_labels=['Did not survive', 'Survived'],
    colorbar=False, ax=ax,
)
ax.set_title('Logistic Regression — confusion matrix')
plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 1 — Reading the confusion matrix</b>
# <ul>
#   <li>
#     Read the four values (TN, FP, FN, TP) from the confusion matrix printed
#     above. In this context (predicting Titanic survival), which type of error —
#     FP or FN — do you think is more costly, and why?
#   </li>
#   <li>
#     Now train a <code>RandomForestClassifier(n_estimators=100, random_state=42)</code>
#     inside the same preprocessing pipeline and display its confusion matrix.
#     Where does the random forest make fewer mistakes than logistic regression?
#   </li>
# </ul>
# </div>

# %%
# Your code here — random forest pipeline and confusion matrix


# %% [markdown]
# ---
# ## 3. Precision, recall, and F1
#
# Three metrics derived directly from the confusion matrix capture the types of
# error in a more actionable way than raw accuracy.
#
# $$\text{Precision} = \frac{TP}{TP + FP}
# \qquad \text{"When I say survivor, how often am I right?"}$$
#
# $$\text{Recall} = \frac{TP}{TP + FN}
# \qquad \text{"Of all true survivors, how many did I find?"}$$
#
# $$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
# \qquad \text{Harmonic mean — punishes extreme imbalance between the two}$$
#
# **Precision–recall trade-off.** Increasing the classification threshold makes
# the model more *conservative* — it predicts "survived" less often, raising
# precision but lowering recall. Decreasing it has the opposite effect.
# The `classification_report` function prints all three metrics at once:

# %%
from sklearn.metrics import classification_report

y_pred_lr = lr_pipeline.predict(X_test)

print("=== Logistic Regression ===")
print(classification_report(
    y_test, y_pred_lr,
    target_names=['Did not survive', 'Survived'],
))

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 2 — Precision, recall, and the cost of errors</b>
# <ul>
#   <li>
#     Print the <code>classification_report</code> for your random forest pipeline
#     from Exercise 1. Which classifier has higher recall for the "Survived" class?
#     Which has higher precision?
#   </li>
#   <li>
#     Imagine you are designing a Titanic early-warning system that alerts the crew
#     which passengers need rescuing. Should you optimise for precision or for recall
#     of the "Survived" class? Justify your answer, and identify which of the two
#     pipelines you would deploy.
#   </li>
#   <li>
#     <b>Threshold experiment.</b> Logistic regression exposes
#     <code>predict_proba</code>: probabilities for each class.
#     Instead of the default threshold of 0.5, set it to 0.35 using the
#     snippet below and observe how precision and recall change.
#     <pre>
# probas   = lr_pipeline.predict_proba(X_test)[:, 1]
# y_pred_35 = (probas >= 0.35).astype(int)
#     </pre>
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# ---
# ## 4. The ROC curve and AUC-ROC
#
# The confusion matrix and F1 score evaluate a classifier at a *single* decision
# threshold. The **Receiver Operating Characteristic (ROC) curve** sweeps over
# all possible thresholds and plots:
#
# - **True Positive Rate (Recall)** on the y-axis — fraction of survivors correctly identified
# - **False Positive Rate** on the x-axis — fraction of non-survivors incorrectly labelled
#   as survivors
#
# A perfect classifier traces the top-left corner (FPR = 0, TPR = 1).
# A random classifier traces the diagonal (FPR = TPR at every threshold).
#
# The **Area Under the Curve (AUC-ROC)** summarises the curve as a single number:
# 0.5 = random, 1.0 = perfect.

# %%
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(6, 5))

RocCurveDisplay.from_estimator(
    lr_pipeline, X_test, y_test,
    name='Logistic Regression', ax=ax,
)

# Diagonal reference line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
ax.set_title('ROC curve — Titanic survival')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 3 — Comparing classifiers with the ROC curve</b>
# <ul>
#   <li>
#     On a single figure, plot the ROC curves for:
#     <ol>
#       <li>The Logistic Regression pipeline (already fitted)</li>
#       <li>Your Random Forest pipeline from Exercise 1</li>
#       <li>A <code>LinearSVC</code> pipeline (use <code>decision_function</code>
#           instead of <code>predict_proba</code> — pass it to
#           <code>RocCurveDisplay.from_estimator</code> directly; sklearn handles this
#           automatically)</li>
#       <li>The <code>DummyClassifier</code></li>
#     </ol>
#     Include the AUC value in each curve's label (use the <code>name=</code>
#     argument of <code>RocCurveDisplay.from_estimator</code>).
#   </li>
#   <li>
#     Which classifier dominates the others across all thresholds?
#     Is the ranking by AUC the same as the ranking by F1?
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# ---
# ## 5. Choosing the right scoring metric for cross-validation
#
# The `scoring` argument of `cross_val_score` accepts any string from
# `sklearn.metrics.SCORERS`. The choice of metric determines which model
# wins a fair comparison — and different metrics can produce different winners.
#
# Common classification scoring strings:
# | String              | What it measures                         |
# |---------------------|------------------------------------------|
# | `'accuracy'`        | Fraction of correct predictions          |
# | `'f1'`              | F1 for the positive class                |
# | `'roc_auc'`         | AUC-ROC (threshold-independent)          |
# | `'average_precision'` | Area under precision-recall curve      |
# | `'balanced_accuracy'` | Mean recall per class (good for imbalance) |

# %%
rf_pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(n_estimators=100, random_state=42),
)

scoring_metrics = ['accuracy', 'f1', 'roc_auc', 'balanced_accuracy']
results = {}

for metric in scoring_metrics:
    lr_scores = cross_val_score(lr_pipeline, features, target, cv=5, scoring=metric)
    rf_scores = cross_val_score(rf_pipeline, features, target, cv=5, scoring=metric)
    results[metric] = {
        'Logistic Regression': lr_scores.mean(),
        'Random Forest':       rf_scores.mean(),
    }

df_results = pd.DataFrame(results).T
print(df_results.round(3).to_string())

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 4 — Does the winner depend on the metric?</b>
# <ul>
#   <li>
#     Look at the table above. For which metrics does Logistic Regression beat
#     Random Forest, and for which does Random Forest win?
#   </li>
#   <li>
#     Add a <code>LinearSVC</code> pipeline to the comparison.
#     Tip: <code>LinearSVC</code> has no <code>predict_proba</code>, so
#     <code>'roc_auc'</code> will fail unless you wrap it with
#     <a href="https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html">
#     <code>CalibratedClassifierCV</code></a>. Import it and try.
#   </li>
#   <li>
#     If you had to deploy one model for a downstream task where false negatives
#     are twice as costly as false positives, which metric would you use to
#     select the model, and which model would you pick?
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# ---
# ## Part 2 — Regression Metrics

# %% [markdown]
# ## 6. Regression metrics
#
# For regression, the equivalent question is: what makes a good prediction error?
# The four most common metrics are:
#
# | Metric | Formula | Sensitivity to outliers |
# |--------|---------|------------------------|
# | **MAE** — Mean Absolute Error | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Low — equal weight to all errors |
# | **RMSE** — Root Mean Squared Error | $\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2}$ | High — large errors dominate |
# | **MAPE** — Mean Absolute Percentage Error | $\frac{100}{n}\sum\left|\frac{y_i-\hat{y}_i}{y_i}\right|$ | Percentage-scale, but unstable near zero |
# | **R²** — Coefficient of determination | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | Scale-free: 1 = perfect, 0 = baseline mean |
#
# RMSE and MAE are in the same units as the target (here, US dollars).
# **When RMSE >> MAE it is a signal that the model has a few very large errors.**

# %%
# Load Ames housing and build a reference Ridge pipeline
ames = load_ames_housing()
target_ames   = ames['SalePrice']
features_ames = ames.drop(columns='SalePrice')

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    features_ames, target_ames, test_size=0.20, random_state=42
)

# Preprocessor for Ames: numeric only for simplicity
numeric_cols = features_ames.select_dtypes('number').columns.tolist()
ames_preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_cols),
    remainder='drop',
)

ridge_pipeline = make_pipeline(
    ames_preprocessor,
    Ridge(alpha=10.0),
)
ridge_pipeline.fit(X_train_a, y_train_a)
y_pred_ridge = ridge_pipeline.predict(X_test_a)

# %% [markdown]
# ### Computing regression metrics

# %%
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

mae  = mean_absolute_error(y_test_a, y_pred_ridge)
rmse = root_mean_squared_error(y_test_a, y_pred_ridge)
r2   = r2_score(y_test_a, y_pred_ridge)
mape = np.mean(np.abs((y_test_a - y_pred_ridge) / y_test_a)) * 100

print(f"Ridge (numeric features only)")
print(f"  MAE  : ${mae:,.0f}")
print(f"  RMSE : ${rmse:,.0f}  ← larger than MAE because of a few big errors")
print(f"  MAPE : {mape:.1f}%")
print(f"  R²   : {r2:.3f}")
print(f"\nRMSE / MAE ratio : {rmse / mae:.2f}  (ratio = 1 means all errors equal size)")

# %% [markdown]
# ### Residual plot
#
# A **residual plot** (predicted vs. error) is the most informative diagnostic
# for regression. Systematic patterns in the residuals reveal model failures that
# single-number metrics hide.

# %%
residuals = y_test_a - y_pred_ridge

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residuals vs predicted
axes[0].scatter(y_pred_ridge, residuals, alpha=0.4, s=12)
axes[0].axhline(0, color='k', linewidth=1)
axes[0].set_xlabel('Predicted SalePrice ($)')
axes[0].set_ylabel('Residual ($)')
axes[0].set_title('Residuals vs Predicted')

# Distribution of residuals
axes[1].hist(residuals, bins=50, edgecolor='white')
axes[1].axvline(0, color='k', linewidth=1)
axes[1].set_xlabel('Residual ($)')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of residuals')

plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 5 — Regression metrics and outlier sensitivity</b>
# <ul>
#   <li>
#     <b>Compare two models.</b> Train a
#     <code>RandomForestRegressor(n_estimators=100, random_state=42)</code>
#     inside the same preprocessing pipeline. Compute MAE, RMSE, MAPE, and R²
#     for both Ridge and the Random Forest. Which is better, and does the ranking
#     depend on the metric?
#   </li>
#   <li>
#     <b>Identify the worst predictions.</b> Find the 10 test houses with the
#     largest absolute residual. Print their <code>SalePrice</code> and
#     <code>GrLivArea</code> (above-ground living area in sq ft). Are they
#     particularly cheap, expensive, or unusual in size?
#   </li>
#   <li>
#     <b>Quantify outlier impact.</b> Recompute RMSE and MAE for the Ridge model,
#     but exclude the 10 worst predictions identified above. How much does RMSE
#     change relative to MAE? What does this tell you about which metric to use
#     when your dataset contains luxury or distressed properties?
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 6 — Scoring in cross-validation for regression</b>
# <ul>
#   <li>
#     Use <code>cross_val_score</code> with <code>scoring='neg_mean_absolute_error'</code>
#     and <code>scoring='r2'</code> to compare Ridge and
#     Random Forest across 5 folds. Note that sklearn negates MAE to follow
#     the convention that higher = better; remember to negate the result back
#     before reporting.
#   </li>
#   <li>
#     Does the ranking (Ridge vs. Forest) change between the two metrics?
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# ---
# ## Part 3 — Class Imbalance

# %% [markdown]
# ## 7. The accuracy trap with imbalanced data
#
# Real-world classification problems are frequently imbalanced: fraud detection
# (< 0.1 % fraud), disease screening (rare positive cases), defect detection in
# manufacturing. The Titanic imbalance (38 / 62 %) is mild. The following
# simulation shows what happens with a **severe** imbalance.

# %%
from sklearn.datasets import make_classification

X_imb, y_imb = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=4,
    weights=[0.97, 0.03],   # 97 % negative, 3 % positive
    flip_y=0,
    random_state=42,
)

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_imb, y_imb, test_size=0.20, random_state=42, stratify=y_imb
)

print(f"Positive class proportion in test set : {y_test_i.mean()*100:.1f} %")

dummy_i = DummyClassifier(strategy='most_frequent')
dummy_i.fit(X_train_i, y_train_i)
print(f"Dummy classifier accuracy             : {dummy_i.score(X_test_i, y_test_i):.3f}")

# %% [markdown]
# A trivial model that always predicts "no fraud" achieves **97 % accuracy**
# while detecting **zero** fraud cases. Let us see what a default
# `LogisticRegression` does:

# %%
from sklearn.metrics import f1_score

lr_imb = LogisticRegression(max_iter=1000, random_state=42)
lr_imb.fit(X_train_i, y_train_i)
y_pred_imb = lr_imb.predict(X_test_i)

print("=== Default LogisticRegression ===")
print(f"Accuracy : {lr_imb.score(X_test_i, y_test_i):.3f}")
print(f"F1 (positive class) : {f1_score(y_test_i, y_pred_imb):.3f}")
print()
print(classification_report(y_test_i, y_pred_imb,
                             target_names=['Negative', 'Positive']))

# %% [markdown]
# ### Fixing imbalance with `class_weight='balanced'`
#
# The `class_weight='balanced'` parameter tells the model to weight each sample
# by the inverse frequency of its class. Minority-class samples count more in the
# loss, pushing the model to pay more attention to them.

# %%
lr_balanced = LogisticRegression(
    max_iter=1000, random_state=42,
    class_weight='balanced',
)
lr_balanced.fit(X_train_i, y_train_i)
y_pred_bal = lr_balanced.predict(X_test_i)

print("=== LogisticRegression(class_weight='balanced') ===")
print(f"Accuracy : {lr_balanced.score(X_test_i, y_test_i):.3f}")
print(f"F1 (positive class) : {f1_score(y_test_i, y_pred_bal):.3f}")
print()
print(classification_report(y_test_i, y_pred_bal,
                             target_names=['Negative', 'Positive']))

# %% [markdown]
# ### The precision-recall curve
#
# For severely imbalanced data the ROC curve can be misleadingly optimistic
# because FPR (the x-axis) stays small even when the model makes many FP errors
# relative to the positive class. The **precision-recall (PR) curve** is more
# informative: it plots precision vs. recall across all thresholds, and its area
# (Average Precision, AP) is a better summary statistic for imbalanced problems.

# %%
from sklearn.metrics import PrecisionRecallDisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ROC
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(
    lr_imb, X_test_i, y_test_i, name='Default LR', ax=axes[0])
RocCurveDisplay.from_estimator(
    lr_balanced, X_test_i, y_test_i, name='Balanced LR', ax=axes[0])
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_title('ROC curve')

# Precision-Recall
PrecisionRecallDisplay.from_estimator(
    lr_imb, X_test_i, y_test_i, name='Default LR', ax=axes[1])
PrecisionRecallDisplay.from_estimator(
    lr_balanced, X_test_i, y_test_i, name='Balanced LR', ax=axes[1])
no_skill = y_test_i.mean()
axes[1].axhline(no_skill, color='k', linestyle='--',
                label=f'No-skill (AP = {no_skill:.2f})')
axes[1].set_title('Precision-Recall curve')
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 7 — Diagnosing and correcting class imbalance</b>
# <ul>
#   <li>
#     <b>Side-by-side confusion matrices.</b> Display the confusion matrix for the
#     default <code>LogisticRegression</code> and the balanced version on the
#     <em>same figure</em> (use <code>fig, axes = plt.subplots(1, 2)</code>).
#     How does the number of detected positives change?
#   </li>
#   <li>
#     <b>Threshold sweep.</b>
#     For the default model, use <code>predict_proba</code> to get the positive-class
#     probability. Sweep thresholds from 0.01 to 0.99 in steps of 0.01, and for
#     each threshold compute precision and recall. Plot them on the same axis as a
#     function of the threshold. At what threshold does F1 peak?
#     <br>Hint: <code>from sklearn.metrics import precision_recall_curve</code>
#     already returns (precision, recall, thresholds) without a manual loop.
#   </li>
#   <li>
#     <b>Go back to Titanic.</b> The Titanic dataset has a 38/62 imbalance.
#     Refit the logistic regression pipeline with
#     <code>LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)</code>.
#     Compare F1, AUC-ROC, and balanced accuracy against the default model.
#     Does balancing help on mild imbalance as much as on severe imbalance?
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# ---
# ## 8. Putting it all together: a metric selection guide
#
# | Situation | Recommended metric |
# |-----------|-------------------|
# | Balanced classes, all errors equal cost | Accuracy |
# | Imbalanced classes | Balanced accuracy, F1, AUC-ROC |
# | False negatives costly (disease, fraud) | Recall, Average Precision |
# | False positives costly (spam filter, alert fatigue) | Precision |
# | Threshold-independent comparison | AUC-ROC |
# | Severe imbalance (< 5 % positive) | Average Precision (PR-AUC) |
# | Regression, all errors matter equally | MAE |
# | Regression, large errors matter more | RMSE |
# | Regression, need scale-free comparison | R² |
#
# <div class="alert alert-success">
#
# <b>EXERCISE 8 (open-ended) — Choose your metric</b>
# <ul>
#   <li>
#     For each scenario below, state which metric you would use as the
#     <em>primary</em> optimisation target in <code>cross_val_score</code>,
#     and briefly justify your choice:
#     <ol>
#       <li>Predicting whether a bank transaction is fraudulent (0.05 % fraud rate).
#           The bank loses $5 000 per undetected fraud but incurs a $50 investigation
#           cost per false alarm.</li>
#       <li>Predicting house sale price for an automated valuation model used in
#           mortgage underwriting, where grossly overvaluing a property (large
#           positive error) is more dangerous than undervaluing it.</li>
#       <li>Predicting which patients in a population screening programme carry a
#           rare genetic variant (0.3 % prevalence). A missed case leads to lack of
#           treatment; a false positive leads to a confirmatory test (low cost).</li>
#       <li>Classifying customer support emails into 12 categories of roughly equal
#           frequency, where routing to the wrong team has similar costs for any pair
#           of categories.</li>
#     </ol>
#   </li>
# </ul>
# </div>

# %%
# Write your answers as comments below:
# Scenario 1:
# Scenario 2:
# Scenario 3:
# Scenario 4:

# %% [markdown]
# ---
# ## Summary
#
# In this notebook you:
# - Demonstrated that **accuracy is misleading** on imbalanced datasets:
#   a classifier that never predicts "survived" achieves 62 % on Titanic.
# - Used the **confusion matrix** to decompose errors into FP and FN, and
#   chose which error type matters more depending on the problem.
# - Computed **precision, recall, F1**, and understood the precision-recall
#   trade-off and how to control it via the decision threshold.
# - Plotted **ROC curves** and computed AUC-ROC to compare classifiers
#   independent of any single threshold.
# - Chose the right `scoring` parameter for `cross_val_score` and saw that
#   **model rankings can differ across metrics**.
# - Computed regression metrics (MAE, RMSE, MAPE, R²) and established that
#   **RMSE >> MAE signals a few large errors** that deserve investigation.
# - Diagnosed **class imbalance** (97/3 split) and applied
#   `class_weight='balanced'` and the **precision-recall curve** as the
#   appropriate tool for severely imbalanced problems.
#
# **Next notebook:** *Data Leakage* — now that you know which metric to trust,
# the next question is whether your cross-validation estimate of that metric
# is itself trustworthy.
