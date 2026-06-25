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
%pip install -q git+https://github.com/demianw/Xed.git

# %% id="SbWLzPLpwQ5l"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %% [markdown] id="nsb_vZjOwQ5l"
# %% [markdown]
# **Prerequisites:** complete *notebook_titanic_survival_prediction* first.
#
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Translate a verbal socio-economic hypothesis into a falsifiable ML question.
# 2. Build a targeted feature subset and classification pipeline to test the hypothesis.
# 3. Interpret classifier coefficients and confusion matrices as evidence for/against the hypothesis.

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

# %% [markdown] id="uL26we-owQ5m"
# Using the `pairplot` of `seaborn` on the `Age`, `Pclass`, `Fare`, `Sex`, and `Survived` columns, identify some intuitions regarding the correlation between the survival and the features. Make some plots to confirm your intuition.

# %% id="ux7vtBBIwQ5m"

# %% [markdown] id="QJTPTQmPwQ5m"
# ## Section 2. Testing a Hypothesis
#
# A socio-economic claim about the Titanic disaster is:
#
# > *"The rich were saved more often than the poor — and wealthy men were
# > particularly likely to pay for their own survival at the expense of others."*
#
# Use the classification pipeline from the main Titanic notebook to test this.
#
# ### Question 1 — Operationalise the hypothesis
#
# Identify which features in the dataset capture *wealth* and *gender*.
# Build a pipeline that uses only those features to predict survival.
# What is the accuracy? What is the chance-level baseline?
#
# ### Question 2 — Examine the coefficients
#
# Fit a `LogisticRegression` on the selected features. Look at the fitted
# coefficients. Which direction and magnitude does each feature contribute?
# Does the sign of the coefficient for `Pclass` support the hypothesis?
#
# ### Question 3 — Stratify by gender
#
# Split the test set into male and female passengers. For each group,
# compute the confusion matrix. Do wealthy men show a higher survival rate
# than their poorer male counterparts? Does the data support or refute
# the "greedy rich men" sub-claim?
#
# ### Question 4 — Full model comparison
#
# Compare the hypothesis-driven model (only wealth + gender features) with
# the full model from the main notebook. Which has higher accuracy?
# What does this tell you about the information not captured by wealth alone?

# %% id="RTX8DeDgXNck"
