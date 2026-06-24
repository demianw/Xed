# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3.8.13
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data visualization and exploration with Pandas
#
# Author: Demian Wassermann, based on Alexandre Gramfort's

# %% [markdown]
# ### Data:
#
# MovieLens 1M Data Set contain les grades given to movies by users on the Movielens website.
#
# The data are available at:
#
# https://moodle.polytechnique.fr/mod/folder/view.php?id=54149
#
# and comes from:
#
# http://grouplens.org/datasets/movielens/

# %% [markdown]
# %% [markdown]
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Load and merge multi-table datasets with Pandas `merge`.
# 2. Use `groupby` and aggregations to compute summary statistics.
# 3. Visualise distributions and relationships with seaborn.
# 4. Formulate data-driven questions and answer them from a real dataset.

# ### Import necessary packages

# %%
# %matplotlib inline

# %%
import pandas as pd
import numpy as np
import seaborn as sns

# %% [markdown]

# %% [markdown]
# ### Downloading the MovieLens 1M dataset

# %%
import os, urllib.request, zipfile

if not os.path.exists("ml-1m"):
    print("Downloading MovieLens 1M...")
    urllib.request.urlretrieve(
        "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ml-1m.zip"
    )
    with zipfile.ZipFile("ml-1m.zip") as zf:
        zf.extractall(".")
    os.remove("ml-1m.zip")
    print("Done.")
else:
    print("Dataset already present.")

# ### Load the `users` data as a Pandas DataFrame

# %%
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('ml-1m/users.dat', sep='::', header=None, names=unames, engine='python')

# %%
users.head()

# %% [markdown]
# ### Read the "rating"

# %%
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')

# %%
ratings.head(10)

# %% [markdown]
# ### Read the movies

# %%
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(
    'ml-1m/movies.dat', sep='::',
    header=None, names=mnames, engine='python',
    encoding="utf8"
)

# %%
movies.head(10)

# %% [markdown]
# ### Let's merge everything as a single DataFrame

# %%
data = pd.merge(pd.merge(ratings, users), movies)

# %%
data.head()

# %% [markdown]
# # Let's explore the data

# %% [markdown]
# ### Question 1
#
# How many movies have a grade higher than 4.5 ?
# Is there a difference between Male or Females?

# %%

# %% [markdown]
# ### Question 2
#
# How many movies have a median grade higher than 4,5 among the men older than 30 years? And among the women older than 30?

# %%

# %% [markdown]
# ### Question 3a
#
# What are the most popular movies?
#
# Hint: use the `DataFrame.nlargest` method.

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# ### Question 3b
#
# What are the most popular movies among the movies that have at least 30 grades? 

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# ### Question 3c
#
# What is the movie with the highest number of ratings?

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# # Data Visualization

# %%
# %matplotlib inline 

# %% [markdown]
# ### Question 4
#
# Show the histogram of the ratings.

# %%

# %% [markdown]
# ### Question 5
#
# Show the histogram of the number of grades obtained for each movie.

# %%

# %% [markdown]
# ### Question 6
#
# Show the histogram of the mean grade for each movie.
#
# Does the distribution of the grade depend on the gender of the user?

# %%

# %% [markdown]
# ### Question 7
#
# Show the histogram of grades among the movies that have been graded at least 30 times.

# %%

# %% [markdown]
# ### Question 8
#
# Show as "scatter plot" the mean grades for the men vs the grades of the women.
#
# Now restrict the plot to the movies with at least 100 grades.

# %%

# %% [markdown]
# # Question 9
#
# Find positive or negative evidence for data being clustered.

# %%
