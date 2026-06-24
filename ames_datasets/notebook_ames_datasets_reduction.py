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

# %% id="MaYbUovVL1ua"
# %pip install -q pandas==2.2.3 seaborn==0.13.2 scikit-learn==1.6.1

# %% id="YMnzmNN3L1ub"
import os
if not os.path.exists('Xed'):
    os.system('git clone --depth=1 https://github.com/demianw/Xed.git')
if 'ames_datasets' not in os.getcwd():
    os.chdir('Xed/ames_datasets')

# %% [markdown] id="3m0k94R4L1ub"
# # Exploring Real Estate Sales Prices — Dimensionality Reduction

# %% id="vPVVzLXKL1uc"
# %matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown] id="sVui2oEEL1uc"
#

# %% [markdown] id="4KOtAe7dL1uc"
# %% [markdown]
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Apply PCA and Kernel PCA to high-dimensional tabular data inside a scikit-learn pipeline.
# 2. Choose the number of components using explained-variance plots and prediction error.
# 3. Identify the most predictive directions in feature space via permutation importance.
# 4. Compare linear (PCA) and non-linear (Kernel PCA) dimensionality reduction for regression.

# ## 1. Exploration

# %% [markdown] id="Od8ovn9zL1uc"
# ### Question 1
#
# Load the Ames dataset using `pandas`. It is located in `datasets/ames_housing.csv`. Using the function `head()` and `info()`, which issues do you identify which need to be noted before to learn a machine learning model.
#
# The dataset is described in https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

# %% id="bAN_fZy5L1uc" outputId="37ef9d05-88ae-4758-8200-623033f813e9"
data = pd.read_csv('../datasets/ames_housing.csv')
data.head()

# %% [markdown] id="vKc38RPAL1uc"
# ### Question 2
#
# - Identify the target variable: `SalePrice`. What are its distributional characteristics?
# - Which variables contain the most missing values?

# %% id="QhLuZz0ML1uc"

# %% [markdown] id="FiwPm29GL1ud"
# ## Question 3
# Split the data into features and target variables.
# Then, the data into a model selection, sample and a model evaluation sample. Use `sklearn.model_selection.train_test_split`.
# Use a 20% ratio.

# %% id="YdWRoFbiL1ud"

# %% [markdown] id="6wdy98o3L1ud"
# ## Question 4
# Extract the columns with numerical data using `selection_features.select_dtypes("number")`. Examine their distributions, through histograms. What issues do you identify?

# %% id="F85j4ePeL1ud"

# %% [markdown] id="IO58hkFEL1ud"
# # Section 2: Implement a dimensionality reduction system using only the numerical variables

# %% [markdown] id="8t-b8PoKL1ud"
# ### Question 1
# Use a Column Transformer to _just select_ the numerical variables. Build a dimensionality reduction pipeline using `sklearn.decomposition.PCA`
#
# For this we will
# * build the column transformer, takig into account missing values and standardization.
# * build the machine learning pipeline
#
# How many components do you need to express 90% of the variance of the data? Why?
#
# * Use if your pipeline is called `pipeline` use `pipeline[-1][1].explained_variance_ratio_` to figure out how much variance explains each component of PCA
#
# Finally
#
# * Use the `fit_transform` method of the pipeline to extract the transformed features in low dimensions and plot a scatterplot with `plt.scatter` of the first two dimensions, which are columns in the output. How do these plots look?

# %% id="LJdgdW6ML1ud"
from sklearn.decomposition import PCA
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# %% id="qkx9qkGdL1ud"

# %% [markdown] id="NC09AKDqL1ud"
# ### Question 2
# Repeat the same exercise but using KernelPCA, a non-linear version of PCA.
#

# %% id="9epACX4uL1ud"
from sklearn.decomposition import KernelPCA


# %% [markdown] id="-ZFj5-xyL1ud"
# ### Question 3
# Use the PCA and KernelPCA dimensionality reduction to implement a regression syste for the house prices. How many components you need to have an error in the estimation of the price of less than 15%? How much data do you need to train these models?

# %% id="F-4LXrGoL1ud"

# %% [markdown] id="tN89pQ7YL1ue"
# ### Question 4
#
# Use permutation importance to analyze which are the most important predictive variables.

# %% id="d_hPfbfCL1ue"

# %% [markdown]
# ---
# ## Summary
#
# In this notebook you:
# - Reduced the Ames housing feature space with PCA and Kernel PCA inside a full pipeline.
# - Determined the minimum number of components for <15 % price-prediction error.
# - Compared the linear and non-linear representations using learning curves.
# - Ranked original features by permutation importance in the reduced space.
#
# **Next notebook:** *Foundation Models for Tabular Data* — see how Prior-data Fitted
# Networks (TabPFN) compare to classical pipelines on the same task.
