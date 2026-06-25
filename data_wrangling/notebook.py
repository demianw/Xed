# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

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

# %%

# %% [markdown]
# # Data wrangling
# %% [markdown]
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Create and inspect pandas `DataFrame` and `Series` objects.
# 2. Select data by label (`.loc`), position (`.iloc`), and boolean masks.
# 3. Compute descriptive statistics and grouped aggregations (`groupby`).
# 4. Merge DataFrames from multiple sources.
# 5. Work with time-indexed data using `resample` and datetime indexing.

# %% [markdown]
# This notebook is adapted from Joris Van den Bossche tutorial:
#
# * https://github.com/paris-saclay-cds/python-workshop/blob/master/Day_1_Scientific_Python/02-pandas_introduction.ipynb

# %%
import numpy as np
import pandas as pd

from xed.datasets import (
    load_titanic,
    load_no2,
    load_french_referendum,
    load_french_departments,
    load_french_regions,
    fetch_french_geojson,
)
import matplotlib.pyplot as plt

pd.options.display.max_rows = 8

# %% [markdown]
# ## 1. Pandas: data analysis in python
#
# For data-intensive work in Python the [Pandas](http://pandas.pydata.org) library has become essential.
#
# **What is `pandas`?**
#
# * Pandas can be thought of as *NumPy arrays with labels* for rows and columns, and better support for heterogeneous data types, but it's also much, much more than that.
# * Pandas can also be thought of as `R`'s `data.frame` in Python.
# * Powerful for working with missing data, working with time series data, for reading and writing your data, for reshaping, grouping, merging your data, ...
#
# It's documentation: http://pandas.pydata.org/pandas-docs/stable/
#
#
# **When do you need pandas?**
#
# When working with **tabular or structured data** (like R dataframe, SQL table, Excel spreadsheet, ...):
#
# - Import data
# - Clean up messy data
# - Explore data, gain insight into data
# - Process and prepare your data for analysis
# - Analyse your data (together with scikit-learn, statsmodels, ...)
#
# <div class="alert alert-warning">
# <b>ATTENTION!</b>: <br><br>
#
# Pandas is great for working with heterogeneous and tabular 1D/2D data, but not all types of data fit in such structures!
# <ul>
# <li>When working with array data (e.g. images, numerical algorithms): just stick with numpy</li>
# <li>When working with multidimensional labeled data (e.g. climate data): have a look at [xarray](http://xarray.pydata.org/en/stable/)</li>
# </ul>
# </div>

# %% [markdown]
# ## 2. The pandas data structures: `DataFrame` and `Series`
#
# ### 2.1 The 2D table: pandas `DataFrame`
#
# A `DataFrame` is a **tablular data structure** (multi-dimensional object to hold labeled data) comprised of rows and columns, akin to a spreadsheet, database table, or R's data.frame object. You can think of it as multiple Series object which share the same index.
#
#
# <img align="left" width=50% src="./schema-dataframe.svg">

# %% [markdown]
# We can create a pandas Dataframe and specify the index and columns to use.

# %%
data = {'country': ['Belgium', 'France', 'Germany', 'Netherlands', 'United Kingdom'],
        'population': [11.3, 64.3, 81.3, 16.9, 64.9],
        'area': [30510, 671308, 357050, 41526, 244820],
        'capital': ['Brussels', 'Paris', 'Berlin', 'Amsterdam', 'London']}
df_countries = pd.DataFrame(data)
df_countries

# %% [markdown]
# We can check that we are manipulating a Pandas DataFrame

# %%
type(df_countries)

# %% [markdown]
# As previously mentioned, the dataframe stores information regarding the column and index information.

# %%
df_countries.columns

# %%
df_countries.index

# %% [markdown]
# You can get an overview of the information of a dataframe using the `info()` method:

# %%
df_countries.info()

# %% [markdown]
# An information which is quite useful is related to the data type.
#
# It is important to know that machine learning algorithms are based on mathematics and algebra. Thus, these algorithms expect numerical data.
#
# Pandas allows to read, manipulate, explore, and transform heterogeneous data to numerical data.

# %%
df_countries.dtypes

# %% [markdown]
# #### Exercise

# %% [markdown]
# We will define a set of 1D NumPy arrays containing the data that we will work with.

# %%
country_name = ['Austria', 'Iran, Islamic Rep.', 'France']
country_code = ['AUT', 'IRN', 'FRA']
gdp_2015 = [1349034029453.37, 385874474398.59, 2438207896251.84]
gdp_2017 = [1532397555.55556, 439513511620.591,2582501307216.42]

# %% [markdown]
# * Create a Python dictionary where the keys will be the name of the columns and the values will be the corresponding Python list.

# %%
# # %cat solutions/02_solutions.py

# %% [markdown]
# * Use the same procedure (Python dictionary) but specify that the country code should be used as the index. Therefore, check the parameter `index_col` or the method `DataFrame.set_index()`

# %%
# # %cat solutions/03_solutions.py

# %%
# # %cat solutions/04_solutions.py

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 2.2 One-dimensional data: `Series` (a column of a DataFrame)
#
# A Series is a basic holder for **one-dimensional labeled data**.

# %%
df_countries

# %%
df_countries.loc[:, 'population']

# %%
population = df_countries.loc[:, 'population']

# %% [markdown]
# We can check that we manipulate a Pandas Series

# %%
type(population)

# %% [markdown]
# ### 2.3 Data import and export

# %% [markdown] slideshow={"slide_type": "subslide"}
# A wide range of input/output formats are natively supported by pandas:
#
# * CSV, text
# * SQL database
# * Excel
# * HDF5
# * json
# * html
# * pickle
# * sas, stata
# * (parquet)
# * ...

# %%
# pd.read_xxxx

# %%
# df.to_xxxx

# %% [markdown]
# Very powerful csv reader:

# %%
# pd.read_csv?

# %% [markdown]
# Luckily, if we have a well formed csv file, we don't need many of those arguments:

# %%
import os

# %%
df = load_titanic()

# %%
df.head()

# %%
df.info()

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE</b>: Load the Belgian NO₂ air quality dataset into a DataFrame `no2`
# using <code>load_no2()</code> from <code>xed.datasets</code>.
# <br><br>
# Some aspects about the file:
#  <ul>
#   <li>Which separator is used in the file?</li>
#   <li>The second row includes unit information and should be skipped (check `skiprows` keyword)</li>
#   <li>For missing values, it uses the `'n/d'` notation (check `na_values` keyword)</li>
#   <li>We want to parse the 'timestamp' column as datetimes (check the `parse_dates` keyword)</li>
# </ul>
# </div>

# %% clear_cell=true
# # %cat solutions/22_solutions.py

# %%
no2.info()

# %% [markdown]
# ## 3. Selecting and filtering data

# %% [markdown]
# One of pandas' basic features is the labeling of rows and columns, but this makes indexing a bit complex. We now have to distinguish between:
#
# * selection by **label**
# * selection by **position**
#

# %% [markdown]
# ### 3.1 Indexing by label using `.loc`

# %% [markdown]
# We will first select data from the dataframe selecting by **label**.

# %%
data = {'country': ['Belgium', 'France', 'Germany', 'Netherlands', 'United Kingdom'],
        'population': [11.3, 64.3, 81.3, 16.9, 64.9],
        'area': [30510, 671308, 357050, 41526, 244820],
        'capital': ['Brussels', 'Paris', 'Berlin', 'Amsterdam', 'London']}
df_countries = pd.DataFrame(data).set_index('country')
df_countries

# %% [markdown]
# The syntax to select by label is `.loc['row_name', 'col_name']`. Therefore, we can get a row of the dataframe by indicating the name of the index to select.

# %%
df_countries.loc['France', :]

# %% [markdown]
# Similarly, we can get a column of the dataframe by indicating the name of the column.

# %%
df_countries.loc[:, 'area']

# %% [markdown]
# Specifying both index and column name, we will get the intersection of the row and the column.

# %%
df_countries.loc['France', 'area']

# %% [markdown]
# We can get several columns by passing a list of the columns to be selected.

# %%
x = df_countries.loc['France', ['area', 'population']]

# %% [markdown]
# This is the exact same behavior with the index for the rows.

# %%
df_countries.loc[['France', 'Belgium'], ['area', 'population']]

# %% [markdown]
# You can go further and slice a portion of the dataframe.

# %%
df_countries.loc['France':'Netherlands', :]

# %% [markdown]
# Note that in this case, the first and last item of the slice are selected.

# %% [markdown]
# ### 3.2  Indexing by position using `.iloc`

# %% [markdown]
# Sometimes, it is handy to select a portion of the data given the row and column indices number. We can this indexing by **position**.

# %%
data = {'country': ['Belgium', 'France', 'Germany', 'Netherlands', 'United Kingdom'],
        'population': [11.3, 64.3, 81.3, 16.9, 64.9],
        'area': [30510, 671308, 357050, 41526, 244820],
        'capital': ['Brussels', 'Paris', 'Berlin', 'Amsterdam', 'London']}
df_countries = pd.DataFrame(data).set_index('country')
df_countries

# %% [markdown]
# The syntax is similar to `.loc`. It will be `.iloc[row_id, col_id]`. We can get the first row.

# %%
df_countries.iloc[0, :]

# %% [markdown]
# Or the last column.

# %%
df_countries.iloc[:, -1]

# %% [markdown]
# And make the intersections.

# %%
df_countries.iloc[0, -1]

# %% [markdown]
# Passing a list of indices is also working.

# %%
df_countries.iloc[[0, 1], [-2, -1]]

# %% [markdown]
# And we can use slicing as well.

# %%
df_countries.iloc[1:3, 0:2]

# %% [markdown]
# However, be aware that the ending index of the slice is discarded.

# %% [markdown]
# ### 3.3 Use the pandas shortcut

# %%
data = {'country': ['Belgium', 'France', 'Germany', 'Netherlands', 'United Kingdom'],
        'population': [11.3, 64.3, 81.3, 16.9, 64.9],
        'area': [30510, 671308, 357050, 41526, 244820],
        'capital': ['Brussels', 'Paris', 'Berlin', 'Amsterdam', 'London']}
df_countries = pd.DataFrame(data).set_index('country')
df_countries

# %% [markdown]
# Pandas provides a shortcut to select some part of the data.

# %%
df_countries['population']

# %%
df_countries[['area', 'capital']]

# %%
df_countries[2:5]

# %%
df_countries['Germany':'United Kingdom']

# %% [markdown]
# You don't need to use `loc` and `iloc`. The selection rules are:
#
# * Passing a single label or list of labels will select a column or several columns;
# * Passing a slice (label or indices) will select the corresponding rows.
#
# You can always use the systematic indexing to avoid confusion. Use the shortcut at your own risk.

# %% [markdown]
# ### 3.4 Boolean indexing (filtering)

# %% [markdown]
# Often, you want to select rows based on a certain condition. This can be done with 'boolean indexing' (like a where clause in SQL) and comparable to numpy. 
#
# The indexer (or boolean mask) should be 1-dimensional and the same length as the thing being indexed.

# %% run_control={"frozen": false, "read_only": false}
df_countries['population'] > 60

# %%
mask_pop_above_60 = df_countries['population'] > 60

# %% [markdown]
# We can then use this mask to index a series or a DataFrame.

# %%
population = df_countries['population']

# %%
population.loc[mask_pop_above_60]

# %%
population[mask_pop_above_60]

# %%
df_countries.loc[mask_pop_above_60]

# %%
df_countries[~mask_pop_above_60]

# %% [markdown]
# ### 3.5 Exercise

# %%
df = load_titanic()
df.head()

# %% [markdown]
# Select the sub-dataframe for which the men are older than 60 years old. Using the attribute shape, find how many individual correspond to this criteria.

# %%
# # %cat solutions/05_solutions.py

# %% [markdown]
# ## 4. Statistical analysis

# %% [markdown]
# Pandas provides an easy and fast way to explore data. Let's explore the `titanic` data set.

# %%
df = df.set_index('Name')
df.head()

# %% [markdown]
# We will select the `Age` column and compute couple of statistic.

# %%
age = df['Age']
age

# %%
age.mean()

# %%
age.max()

# %%
age.min()

# %%
age.describe()

# %%
age.value_counts()

# %%
age.hist(bins=100)

# %% [markdown]
# ### Exercise

# %% [markdown]
# * What is the maximum Fare that was paid? And the median?

# %%
# # %cat solutions/06_solutions.py

# %%
# # %cat solutions/07_solutions.py

# %% [markdown]
# * Calculate the average survival ratio for passengers (note: the 'Survived' column indicates whether someone survived (1) or not (0)).

# %%
# # %cat solutions/08_solutions.py

# %%
# # %cat solutions/09_solutions.py

# %% [markdown]
# * Select the sub-dataframe for which the men are older than 60 years old.

# %%
# # %cat solutions/10_solutions.py

# %% [markdown]
# * Based on the titanic data set, select all rows for male passengers and calculate the mean age of those passengers. Do the same for the female passengers.

# %%
# # %cat solutions/11_solutions.py

# %%
# # %cat solutions/12_solutions.py

# %% [markdown]
# * Plot the Fare distribution.

# %%
# # %cat solutions/13_solutions.py

# %% [markdown]
# ## 5. The group-by operation

# %% [markdown]
# ### Some 'theory': the groupby operation (split-apply-combine)

# %% run_control={"frozen": false, "read_only": false}
df = pd.DataFrame({'key':['A','B','C','A','B','C','A','B','C'],
                   'data': [0, 5, 10, 5, 10, 15, 10, 15, 20]})
df

# %% [markdown]
# ### 5.1 Recap: aggregating functions

# %% [markdown]
# When analyzing data, you often calculate summary statistics (aggregations like the mean, max, ...). As we have seen before, we can easily calculate such a statistic for a Series or column using one of the many available methods. For example:

# %% run_control={"frozen": false, "read_only": false}
df['data'].sum()

# %% [markdown]
# However, in many cases your data has certain groups in it, and in that case, you may want to calculate this statistic for each of the groups.
#
# For example, in the above dataframe `df`, there is a column 'key' which has three possible values: 'A', 'B' and 'C'. When we want to calculate the sum for each of those groups, we could do the following:

# %% run_control={"frozen": false, "read_only": false}
for key in ['A', 'B', 'C']:
    print(key, df[df['key'] == key]['data'].sum())

# %% [markdown]
# This becomes very verbose when having multiple groups. You could make the above a bit easier by looping over the different values, but still, it is not very convenient to work with.
#
# What we did above, applying a function on different groups, is a "groupby operation", and pandas provides some convenient functionality for this.

# %% [markdown]
# ### 5.2 Groupby: applying functions per group

# %% [markdown] slideshow={"slide_type": "subslide"}
# The "group by" concept: we want to **apply the same function on subsets of your dataframe, based on some key to split the dataframe in subsets**
#
# This operation is also referred to as the "split-apply-combine" operation, involving the following steps:
#
# * **Splitting** the data into groups based on some criteria
# * **Applying** a function to each group independently
# * **Combining** the results into a data structure
#
# <img src="./splitApplyCombine.png">
#
# Similar to SQL `GROUP BY`

# %% [markdown]
# Instead of doing the manual filtering as above
#
#
#     df[df['key'] == "A"].sum()
#     df[df['key'] == "B"].sum()
#     ...
#
# pandas provides the `groupby` method to do exactly this:

# %% run_control={"frozen": false, "read_only": false}
df.groupby('key').sum()

# %% run_control={"frozen": false, "read_only": false} slideshow={"slide_type": "subslide"}
df.groupby('key').aggregate([np.sum, np.median])  # 'sum'

# %% [markdown]
# And many more methods are available. 

# %% run_control={"frozen": false, "read_only": false}
df.groupby('key')['data'].sum()

# %%
for group_name, group_df in df.groupby('key'):
    print(group_name)
    print(group_df)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 5.3 Exercise: Application of the groupby concept on the titanic data

# %% [markdown]
# We go back to the titanic passengers survival data:

# %% run_control={"frozen": false, "read_only": false}
df = load_titanic(
df = df.set_index('Name')

# %% run_control={"frozen": false, "read_only": false}
df.head()

# %% [markdown]
# * Using `groupby()`, calculate the average age for each sex.</li>
#

# %% clear_cell=true run_control={"frozen": false, "read_only": false}
# # %cat solutions/14_solutions.py

# %% [markdown]
# * Using the `groupby()` function, plot the age distribution for each sex.

# %%
# # %cat solutions/15_solutions.py

# %% [markdown]
# * Plot the fare distribution based on the class.

# %%
# # %cat solutions/16_solutions.py

# %% [markdown]
# * Plot the survival rate by class with a bar plot.

# %%
# # %cat solutions/17_solutions.py

# %% [markdown]
# * Compute the survival rate grouping by class and sex. (Hint: you can pass a list to the `groupby` function)

# %%
# # %cat solutions/18_solutions.py

# %% [markdown]
# ## 7. Merging different source of information

# %% [markdown]
# ### 7.1 Simple concatenation

# %%
# series
population = pd.Series({'Germany': 81.3, 'Belgium': 11.3, 'France': 64.3, 
                        'United Kingdom': 64.9, 'Netherlands': 16.9})

# dataframe
data = {'country': ['Belgium', 'France', 'Germany', 'Netherlands', 'United Kingdom'],
        'population': [11.3, 64.3, 81.3, 16.9, 64.9],
        'area': [30510, 671308, 357050, 41526, 244820],
        'capital': ['Brussels', 'Paris', 'Berlin', 'Amsterdam', 'London']}
countries = pd.DataFrame(data)
countries

# %% [markdown]
# Assume we have some similar data as in countries, but for a set of different countries:

# %%
data = {'country': ['Nigeria', 'Rwanda', 'Egypt', 'Morocco', ],
        'population': [182.2, 11.3, 94.3, 34.4],
        'area': [923768, 26338 , 1010408, 710850],
        'capital': ['Abuja', 'Kigali', 'Cairo', 'Rabat']}
countries_africa = pd.DataFrame(data)
countries_africa

# %% [markdown]
# We now want to combine the rows of both datasets:

# %%
pd.concat([countries, countries_africa])

# %% [markdown]
# If we don't want the index to be preserved:

# %%
pd.concat([countries, countries_africa], ignore_index=True)

# %% [markdown]
# When the two dataframes don't have the same set of columns, by default missing values get introduced:

# %%
pd.concat([countries_africa[['country', 'capital']], countries], ignore_index=True, sort=False)

# %% [markdown]
# ## 7.2 Combining columns instead of rows

# %% [markdown]
# Assume we have another DataFrame for the same countries, but with some additional statistics:

# %%
data = {'country': ['Belgium', 'France', 'Netherlands'],
        'GDP': [496477, 2650823, 820726],
        'area': [8.0, 9.9, 5.7]}
country_economics = pd.DataFrame(data).set_index('country')
country_economics

# %%
pd.concat([countries, country_economics], axis=1)

# %% [markdown]
# `pd.concat` matches the different objects based on the index:

# %%
countries2 = countries.set_index('country')

# %%
countries2

# %%
pd.concat([countries2, country_economics], axis=1, sort=False)

# %% [markdown]
# ### 7.3 Dataframe merging 

# %% [markdown]
# Using `pd.concat` above, we combined datasets that had the same columns or the same index values. But, another typical case if where you want to add information of second dataframe to a first one based on one of the columns. That can be done with `pd.merge`.

# %% [markdown]
# Let's look again at the titanic passenger data, but taking a small subset of it to make the example easier to grasp:

# %%
df = load_titanic(
df = df.loc[:9, ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

# %%
df

# %% [markdown]
# Assume we have another dataframe with more information about the 'Embarked' locations:

# %%
locations = pd.DataFrame({'Embarked': ['S', 'C', 'Q', 'N'],
                          'City': ['Southampton', 'Cherbourg', 'Queenstown', 'New York City'],
                          'Country': ['United Kindom', 'France', 'Ireland', 'United States']})

# %%
locations

# %% [markdown]
# We now want to add those columns to the titanic dataframe, for which we can use `pd.merge`, specifying the column on which we want to merge the two datasets:

# %%
pd.merge(df, locations, on='Embarked', how='left')

# %% [markdown]
# In this case we use `how='left'` (a "left join") because we wanted to keep the original rows of df and only add matching values from locations to it. Other options are 'inner', 'outer' and 'right' (see the docs for more on this).

# %% [markdown]
# ## 8. Working with time series data

# %% [markdown]
# ### 8.1 Time series preamble

# %%
no2 = load_no2()

# %% [markdown] slideshow={"slide_type": "fragment"}
# When we ensure the DataFrame has a `DatetimeIndex`, time-series related functionality becomes available:

# %%
no2.index

# %% [markdown] slideshow={"slide_type": "subslide"}
# Indexing a time series works with strings:

# %%
no2["2010-01-01 09:00": "2010-01-01 12:00"]

# %% [markdown] slideshow={"slide_type": "subslide"}
# A nice feature is "partial string" indexing, so you don't need to provide the full datetime string.

# %% [markdown] slideshow={"slide_type": "-"}
# E.g. all data of January up to March 2012:

# %%
no2['2012-01':'2012-03']

# %% [markdown] slideshow={"slide_type": "subslide"}
# Time and date components can be accessed from the index:

# %%
no2.index.hour

# %%
no2.index.year

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 8.2 The power of pandas: `resample`

# %% [markdown]
# A very powerful method is **`resample`: converting the frequency of the time series** (e.g. from hourly to daily data).
#
# Remember the air quality data:

# %%
no2.plot()

# %% [markdown]
# The time series has a frequency of 1 hour. I want to change this to daily:

# %%
no2.head()

# %%
no2.resample('D').mean().head()

# %% [markdown] slideshow={"slide_type": "subslide"}
# Above I take the mean, but as with `groupby` I can also specify other methods:

# %%
no2.resample('D').max().head()

# %% [markdown] slideshow={"slide_type": "skip"}
# The string to specify the new time frequency: http://pandas.pydata.org/pandas-docs/dev/timeseries.html#offset-aliases  
# These strings can also be combined with numbers, eg `'10D'`.

# %% [markdown] slideshow={"slide_type": "subslide"}
# Further exploring the data:

# %%
no2.resample('M').mean().plot() # 'A'

# %% clear_cell=true slideshow={"slide_type": "subslide"}
no2.loc['2009':, 'VERS'].resample('M').agg(['mean', 'median']).plot()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 8.3 Exercise

# %% [markdown]
# The evolution of the yearly averages with, and the overall mean of all stations
#
# * Use `resample` and `plot` to plot the yearly averages for the different stations.
# * The overall mean of all stations can be calculated by taking the mean of the different columns (`.mean(axis=1)`).
#

# %%

# %% [markdown]
#
# ## Further reading
#
# * Pandas documentation: http://pandas.pydata.org/pandas-docs/stable/
#
# * Books
#
#     * "Python for Data Analysis" by Wes McKinney
#     * "Python Data Science Handbook" by Jake VanderPlas
#
# * Tutorials (many good online tutorials!)
#
#   * https://github.com/jorisvandenbossche/pandas-tutorial
#   * https://github.com/brandon-rhodes/pycon-pandas-tutorial
#
# * Tom Augspurger's blog
#
#   * https://tomaugspurger.github.io/modern-1.html
