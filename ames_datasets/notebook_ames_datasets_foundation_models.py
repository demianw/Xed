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
#     name: python3
# ---

# %% id="MaYbUovVL1ua" colab={"base_uri": "https://localhost:8080/"} outputId="3432bdb7-dba2-4bc0-8532-efc982141ed9"

# %% id="YMnzmNN3L1ub" colab={"base_uri": "https://localhost:8080/"} outputId="8c643698-e4bd-46f8-f7ac-f51a1c69fe40"
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

# %% [markdown] id="3m0k94R4L1ub"
# # Exploring Real Estate Sales Prices — Foundation Models

# %% id="vPVVzLXKL1uc"
# %matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
from tabpfn import TabPFNRegressor

import matplotlib.pyplot as plt

# %% [markdown] id="4KOtAe7dL1uc"
# %% [markdown]
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Use TabPFN and TabICL — foundation models for tabular data — as drop-in scikit-learn estimators.
# 2. Compare foundation models to classical pipelines using learning curves.
# 3. Understand why TabPFN has a row-count limit and what that implies for evaluation.

# ## 1. Exploration

# %% [markdown] id="Od8ovn9zL1uc"
# ### Question 1
#
# Load the Ames housing dataset using `pandas` from `datasets/ames_housing.csv`. Using the function `head()` and `info()`, which issues do you identify which need to be noted before to learn a machine learning model.
#
# The dataset is described in https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

# %% id="bAN_fZy5L1uc" outputId="75f7abf0-f9c9-4c74-ea9c-ee3424d1691d" colab={"base_uri": "https://localhost:8080/", "height": 266}
data = load_ames_housing()
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
#
# #### **Careful! We need to use a table of at most 1000 records to not have to use paid Google collab**

# %% id="YdWRoFbiL1ud"

# %% [markdown] id="6wdy98o3L1ud"
# ## Question 4
# Extract the columns with numerical data using `selection_features.select_dtypes("number")`. Examine their distributions, through histograms. What issues do you identify?

# %% id="F85j4ePeL1ud"

# %% [markdown] id="IO58hkFEL1ud"
# # Section 2: Use Foundation models

# %% [markdown] id="8t-b8PoKL1ud"
# ### Question 1
# Use TabPFN `from tabpfn import TabPFNRegressor` to perform the regression.
# Evaluate the performance with cross_val_predict
#
# #### **Careful! We need to use a table of at most 1000 records to not have to use paid Google collab**
#

# %% id="UZvRr_1sMBzb"
import os

# %% id="LJdgdW6ML1ud"
from sklearn.model_selection import cross_val_score

# %% id="KKZgXhJkPOnr"
from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor
from tabpfn.constants import ModelVersion

# %% id="OVNfu6t-M8Gy"
# Use the free foundation model for TabPFN
regressor_pfn = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
regressor_icl = TabICLRegressor()

# %% id="4NNL3tRnSAtX"
from warnings import filterwarnings

# Suppress noisy but harmless convergence warnings from TabPFN's internal torch ops.
# In production code, investigate every warning rather than silencing them.
filterwarnings("ignore", category=UserWarning)

# %% id="jIOVqCppRdPl"

# %% [markdown] id="NC09AKDqL1ud"
# ### Question 2
# Compare using the learning curve with one of the previous pipelines we have made. Remember the 1000 record limitation
#

# %% id="9epACX4uL1ud"

# %% [markdown]
# ---
# ## Summary
#
# In this notebook you:
# - Used TabPFN and TabICL as zero-configuration baselines for tabular regression.
# - Compared their learning curves to the classical pipelines from the previous notebook.
# - Observed the 1 000-row inference limit of the free TabPFN server and its
#   implications for fair benchmarking.
#
# **Key take-away:** foundation models for tabular data can match classical pipelines
# with minimal tuning, but their row-count constraints make them most useful as
# quick baselines, not as production solutions on large datasets.
