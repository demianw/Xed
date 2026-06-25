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

# %% id="KjAyiNwsTLMC"

# %% id="16wUQ_vATLMD"
%pip install -q git+https://github.com/demianw/Xed.git

# %% [markdown] id="CbMfJvqNTLMD"
# # Exploring Real Estate Sales Prices

# %% id="UVwpdGdjTLMD"
%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Identify data quality issues (missing values, skewed distributions) in a real-world tabular dataset.
# 2. Build scikit-learn preprocessing pipelines combining imputation, scaling and encoding.
# 3. Evaluate regression models (Linear, Ridge, Lasso, Random Forest) using learning curves and cross-validation.
# 4. Interpret feature importance via permutation importance.

# %% [markdown]
# ## 1. Exploration

# %% [markdown] id="JhmfH9zqTLME"
# ### Question 1
#
# Load the Ames housing dataset using `pandas`. It is located in `datasets/ames_housing.csv`. Using the function `head()` and `info()`, which issues do you identify which need to be noted before to learn a machine learning model.
#
# The dataset is described in https://jse.amstat.org/v19n3/decock/DataDocumentation.txt

# %% id="kGgXJnP_TLME"
data = load_ames_housing()
data.head()

# %% [markdown] id="LwmvOm7tTLME"
# ### Question 2
#
# - Identify the target variable: `SalePrice`, what is its type? What are its distributional characteristics?
# - Which variables contain the most missing values?

# %% id="QozaVi50TLME"

# %% [markdown] id="FCnyCIpzTLME"
# ## Question 3
# Split the data into features and target variables.
# Then, the data into a model selection, sample and a model evaluation sample. Use `sklearn.model_selection.train_test_split`.
# Use a 20% ratio.

# %% id="6wWnT8_bTLME" outputId="6c0385a6-d314-46f3-9df5-5e9cc7182477"
from sklearn.model_selection import train_test_split

target = data["SalePrice"]
features = data.drop(columns="SalePrice")

selection_features, evaluation_features, selection_target, evaluation_target = train_test_split(
    features, target, test_size=0.2
)
selection_target.shape

# %% [markdown] id="QBf8fNjjTLMF"
# ## Question 4
# Extract the columns with numerical data using `selection_features.select_dtypes("number")`. Examine their distributions, through histograms. What issues do you identify? Then use  `selection_features.select_dtypes("number")` and seaborn's `sns.countplot` to analyze the string variables. Identify data types and issues.

# %% id="McWpsSQ1TLMF"

# %% [markdown] id="CGm4F7wFTLMF"
# # Section 2: Implement a linear regressor using only the numerical variables

# %% [markdown] id="5wXKF68YTLMF"
# ### Question 1
# Use a Column Transformer to _just select_ the numerical variables. Build a linear regressor using `sklearn.linear.LinearRegressor
#
# For this we will
# * build the column transformer
# * build the machine learning pipeline
# * evaluate it through cross-validation (using `cross_vals_score`)
#
# Does it work? Why?

# %% id="gQu-pyIhTLMF"
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# %% [markdown] id="vial1zaXTLMF"
# ### Question 2
# Fix the previous issue using: first drop problematic rows, then use the `SimpleImputer`
#

# %% id="Crs9EYiWTLMF"
from sklearn.impute import SimpleImputer

# %% [markdown] id="Cx33j2lhTLMF"
# ### Question 3
# Now plot the evolution of mean and standard deviations for test sample sizes of 05%, 10%, 20%, 25%, 30%. What do you conclude?
#

# %% [markdown] id="5-g7zmJcTLMF"
# ### Question 4
# Use `sklearn.model_selection.learning_curve` to study the learning curve https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html?highlight=learning_curve

# %% id="KM6nJ0HiTLMF"
from sklearn.model_selection import learning_curve

# %% [markdown] id="DxEp40_GTLMF"
# ### Question 5
# Are there correlations between the features? Explore it through the correlation matrix, and the `sns.pairplot` plotting tool from seaborn (warning, if you plot all variables together it might be slow)

# %% id="sXXTvdOSTLMG"

# %% [markdown] id="_fzoNusHTLMG"
# ### Question 6
# Can we use this correlation to improve the learning curve? This is regularization, let's try ridge regression https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
#
# Plot the learning curve and compare it with the plain linear regression

# %% id="3iGpCt5mTLMG"
from sklearn.linear_model import Ridge

# %% [markdown] id="2Gp9AvC5TLMG"
# ### Question 6.1
# How did you pick your regularization parameter? Use a grid search now. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
#

# %% id="UApq7NSDTLMG"
from sklearn.model_selection import GridSearchCV

# %% [markdown] id="_ioYT-bjTLMG"
# ### Question 7
# Do we need all features? Repeat the previous analysis from Question 6 but with the Lasso which enforces sparsity https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

# %% id="1xm7gylUTLMG"

# %% [markdown] id="fZ-FAFJJTLMG"
# ### Question 8
#
# Now we will repeat the same analysis but with the categorical variables. For which we will use the `OneHotEncoder` and the `OrdinalEncoder`
# * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
#
# and combine them in the preprocessing pipeline in Section 2, Questions 1 and 2.

# %% id="v4cRbDifTLMG"
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# %% [markdown] id="WecnTk7WTLMG"
# ## Question 9
#
# Non linearity! Now use the RandomForestRegressor https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html to fit and predict the data. The two hyper-parameters that you will use are
#
# * n_estimators : with a default of 100 which deals with the uncertainty in the data/algorithm relationship.
# * max_depth : with a no limit as a default which deals with the granularity of the solution.
#
# Use a Grid search cross validation to set the two parameters. Plot the learning curve.

# %% id="Yf1C5ze1TLMG"
from sklearn.ensemble import RandomForestRegressor

# %% [markdown] id="Vp7TWN4xTLMG"
# ## Question 10
#
# We will now use the data to obtain
# Use permutation feature importance to assess which are the most important features in predicting house pricing https://scikit-learn.org/stable/modules/permutation_importance.html
#
# Compare these importances across models.

# %% id="zlGAOQ6KTLMH"
from sklearn.inspection import permutation_importance

# %% [markdown] id="hvMdqT0ETLMH"
# ## Question 11
#
# Pick one of the estimators. Use cross_val_predict to evaluate the quality of the prediction in different cases.
#
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html#sphx-glr-auto-examples-model-selection-plot-cv-predict-py
#
# Cross-val predict will give you for each element in the target, a prediction. Produce a scatterplot between target and prediction, use the trained model and the predictive importance to find the most explanatory variables.

# %% id="1oK4yWtETLMH"
from sklearn.model_selection import cross_val_predict

# %% [markdown] id="FXLoapDjTLMH"
# # Section 3: Interpreting the best model
#
# Now that you have selected a model, answer the following questions using
# the tools introduced in Section 2:
#
# ### Question 1
# Use `cross_val_predict` with your best pipeline to plot predicted vs. true
# `SalePrice`. What does the scatter reveal about systematic prediction errors?
#
# ### Question 2
# Compute permutation importances for the best pipeline on the evaluation set.
# Which five features matter most? Do they align with domain intuition about
# house prices?
#
# ### Question 3
# Identify the five houses with the largest absolute prediction error.
# Inspect their raw feature values — can you hypothesise why the model
# struggled with them?

# %% [markdown]
# ---
# ## Summary
#
# In this notebook you:
# - Explored the Ames housing dataset and diagnosed missing values and skewed features.
# - Built preprocessing pipelines combining `SimpleImputer`, `StandardScaler`,
#   `OneHotEncoder`, and `OrdinalEncoder` inside a `ColumnTransformer`.
# - Compared Linear Regression, Ridge, Lasso, and Random Forest using learning curves.
# - Used `GridSearchCV` to tune regularisation strength.
# - Quantified feature contributions via permutation importance.
#
# **Next notebook:** *Dimensionality Reduction* — apply PCA and Kernel PCA to the
# same dataset and ask how many components are needed to retain predictive power.
