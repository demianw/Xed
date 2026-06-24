# Xed — Cardiff Introductory Data Science: Practical Notebooks

Practical exercises for the **Xed Cardiff** introductory data science course.

## Recommended sequence

| # | Topic | Notebook | Concepts |
|---|-------|----------|----------|
| 1 | **Data Wrangling** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/data_wrangling/notebook.ipynb) | Pandas DataFrames, indexing, groupby, merge, time series |
| 2 | **Regression** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/ames_datasets/notebook_ames_datasets.ipynb) | Linear / Ridge / Lasso / Random Forest, learning curves, feature importance |
| 3 | **Classification** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/titanic_survival_prediction/notebook_titanic_survival_prediction.ipynb) | Logistic Regression, SVC, ColumnTransformer, permutation importance |
| 4 | **Dimensionality Reduction** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/ames_datasets/notebook_ames_datasets_reduction.ipynb) | PCA, Kernel PCA, explained variance |
| 5 | **Cross-Validation** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/cross_validation/notebook_cross_validation.ipynb) | KFold, StratifiedKFold, ShuffleSplit |
| 6 | **⚠ Data Leakage** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/data_leakage/notebook_data_leakage.ipynb) | Preprocessing leakage, feature-selection leakage, TimeSeriesSplit |
| 7 | **Decision Trees & Forests** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/visualizations/trees_and_forests.ipynb) | Decision trees, random forests, GridSearchCV |
| 8 | **Text Classification & LLMs** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/text/text_classification_and_llms.ipynb) | TF-IDF, BERT embeddings, zero-shot prompting with T5 |
| 9 | **Foundation Models** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/ames_datasets/notebook_ames_datasets_foundation_models.ipynb) | TabPFN, TabICL, comparison with classical pipelines |
| — | **Hypothesis Testing** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/titanic_survival_prediction/notebook_titanic_survival_prediction_check_hypothesis.ipynb) | Operationalising a socio-economic hypothesis as an ML question |
| — | **Data Exploration (MovieLens)** | [Open in Colab](https://colab.research.google.com/github/demianw/Xed/blob/improvements/movie%20lens/notebook_movie_lens.ipynb) | Pandas merge, groupby, seaborn visualisation |

## Repository structure

The **`.py` files are the source of truth**.
Each `.ipynb` is generated from the corresponding `.py` using
[jupytext](https://jupytext.readthedocs.io/) and is committed only so that
the Colab links above keep working.

To edit a notebook locally:
```bash
pip install jupytext
# Edit the .py file in any editor or IDE, then sync:
jupytext --sync ames_datasets/notebook_ames_datasets.py
# Or open it as a paired notebook in JupyterLab:
jupytext --to notebook ames_datasets/notebook_ames_datasets.py
jupyter lab ames_datasets/notebook_ames_datasets.ipynb
```

## Bibliography

* Joris Van den Bossche — [Python workshop pandas introduction](https://github.com/paris-saclay-cds/python-workshop)
* Alexandre Gramfort — MovieLens visualisation notebook
* [datascience_starter_course](https://github.com/glemaitre/datascience_starter_course) (submodule, bibliography only)
