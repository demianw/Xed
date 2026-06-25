# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3.8.13 ('neurolang')
#     language: python
#     name: python3
# ---

# %% [markdown] id="ZBWde_EtwQ5i"
# # Exploration of the Titanic data set

# %% colab={"base_uri": "https://localhost:8080/"} id="smzP4B7HwQ5k" outputId="46e588e7-d835-4cde-85ec-0a34dd478af9"

# %% id="QNKUn2b1wQ5l"
import os
import subprocess
import sys

# Install the course package and all pinned dependencies.
# In GitHub Actions CI this step is skipped (pre-installed via pip install -e .[dev]).
if not os.environ.get("CI"):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/demianw/Xed.git"],
        check=True,
    )

# %% id="SbWLzPLpwQ5l"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown] id="nsb_vZjOwQ5l"
# %% [markdown]
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Perform exploratory analysis on a mixed (numerical + categorical) classification dataset.
# 2. Build heterogeneous preprocessing pipelines with `ColumnTransformer`.
# 3. Train and compare Logistic Regression, Linear SVC, and Random Forest classifiers.
# 4. Evaluate models with learning curves and `cross_val_predict`.
# 5. Interpret fitted model coefficients and permutation importances.

# ## Section 1. Exploration

# %% [markdown] id="5eiwRVquwQ5l"
# ### Question 1
#
# Load the titanic using `pandas`. It is located in `datasets/titanic.csv`. Using the function `head()` and `info()`, which issues do you identify which need to be solved before to learn a machine learning model.

# %% id="BUg-__bjwQ5l"
from xed.datasets import load_titanic

data = load_titanic()

# %% [markdown] id="yzsFuEecwQ5l"
# ### Question 2
#
# - By checking the variable `Survived`, is the dataset balanced? What will be the chance level accuracy?
# - What variables contain more missing values?

# %% id="CZwqc55xwQ5m"

# %% [markdown] id="0l3lHLoEwQ5m"
# ### Question 3
#
# Using the `pairplot` of seaborn on the columns `Age`, `Pclass`, `Fare`, `Sex`,
# and `Survived`, identify some visual intuitions about which features correlate
# with survival. Then make targeted bar-plots or box-plots to strengthen two of
# those intuitions quantitatively.

# %% [markdown] id="uL26we-owQ5m"
# Using the `pairplot` of `seaborn` on the `Age`, `Pclass`, `Fare`, `Sex`, and `Survived` columns, identify some intuitions regarding the correlation between the survival and the features. Make some plots to confirm your intuition.

# %% id="ux7vtBBIwQ5m"

# %% [markdown] id="QJTPTQmPwQ5m"
# ## Section 2. Predicting survival

# %% [markdown] id="Iu1g_ZPHwQ5m"
# The titanic dataset is an heterogeneous dataset and it gives the opportunity to show the scikit-learn pipelining features. We will show in this notebook how to make a simple classification pipeline. The aim is to predict or not if a passenger survived the titanic trip.

# %% id="Vvo1xUy_wQ5m"
data = load_titanic()

# %% id="YywzXcT8wQ5m"
data.head()

# %% [markdown] id="BfEPeJpVwQ5m"
# First, we need to split the dataset into 2 arrays: the data array and the classification array.

# %% id="mPN8AHi9wQ5m"
label = data["Survived"]
data = data.drop(columns="Survived")

# %% [markdown] id="yNbqoYE9wQ5m"
# Because the data type in the titanic dataset, we need to specifically have different preprocessing for the continuous and categorical columns. The `ColumnTransformer` of scikit-learn allows to dispatch different preprocessing depending of the columns. Usually, the categorical variable needs to be encoded while the continuous variable can be standardized.

# %% id="hjBSUQQUwQ5n"
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# %% [markdown] id="GKGu1CUywQ5n"
# We are creating three preprocessing:
#
# * an ordinal encoding for the sex;
# * a one hot encoding for the remaining categorical features;
# * and a standardization for the continuous features.
#
# In addition, missing values will be filled up with either the median (for continuous variable) or a constant value (categorical variable).

# %% id="RjwOip6swQ5n"
data.head()

# %% id="P-J87BgCwQ5n"

# %% [markdown] id="XnjK1kAQwQ5n"
# A logistic regression classifier will be used in which the C parameter will be optimized. We will apply a 5-fold cross-validation scheme to estimate the accuracy of the model.

# %% id="AA6lObswwQ5n"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict

# %% id="nI7oJBsBwQ5n"
from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = train_test_split(
    data, label, test_size=0.20, random_state=42
)

# %% [markdown] id="QhH8_fdXwQ5n"
# ### Compare different classification algorithms to predict survival. Specifically through the learning curve and the prediction quality. You can compare
# * Logistic Regression, with a parameter C
# * LinearSVC
# * RandomForestClassifiers

# %% id="KGUPLTk6wQ5n"
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# %% [markdown] id="rrXsuu2BwQ5n"
# # Section 3. Model validation and explanatory capabilities. For each model explore fitted model parameters and try to assess their explanatory capabilities
#
# Bear in mind that if `pipe` is our processing pipeline composed of a preprocessing step and a LinearSVC step:
# * `linear_svc = pipe._final_estimator` extracts the tuple (*model name*, *model class*)
# * `linear_svc` extracts regression class
# * `linear_svc.coef_` are the coefficients of the regressors

# %% id="4dDXtDruwQ5n"
from sklearn.inspection import permutation_importance

# %% [markdown]
# ---
# ## Summary
#
# In this notebook you:
# - Diagnosed class imbalance and missing values in the Titanic dataset.
# - Built a heterogeneous preprocessing pipeline for mixed data types.
# - Compared Logistic Regression, LinearSVC, and Random Forest using learning curves.
# - Extracted and interpreted fitted coefficients and permutation importances.
#
# **Next notebook:** *Hypothesis checking with a classification pipeline* — use the
# same tools to test a specific socio-economic hypothesis about who survived.
