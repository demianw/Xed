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

# %% [markdown]
# # Text Classification and LLMs: From TF-IDF to Zero-Shot Prompting
#
# **Prerequisites:** *Evaluation Metrics*, *Dimensionality Reduction* notebooks.
#
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Represent text as TF-IDF vectors and classify with Logistic Regression.
# 2. Apply dimensionality reduction (PCA, Kernel PCA) to sparse text features.
# 3. Extract dense sentence embeddings with a pre-trained transformer (BERT family).
# 4. Use a generative LLM (T5) for zero-shot classification via prompting.
# 5. Compare TF-IDF, BERT embeddings, and zero-shot prompting on the same task.

# %%
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

rng = np.random.default_rng(42)

# %% [markdown]
# ## (Optional) Hugging Face account setup
#
# The models used in this notebook (`all-mpnet-base-v2`, `flan-t5-small`) are
# **free and open** — authentication is NOT required but avoids throttling.
#
# If you want higher rate limits:
# 1. Create a free account at https://huggingface.co/join
# 2. Generate a read token at https://huggingface.co/settings/tokens
# 3. In Colab: add `HF_TOKEN` to the Secrets tab (🔑), then uncomment below:
#
# ```python
# # from huggingface_hub import login
# # from google.colab import userdata
# # login(token=userdata.get("HF_TOKEN"))
# ```

# %% [markdown]
# ## 1. Loading and exploring the dataset
#
# We will work with the **Rotten Tomatoes** movie-review sentiment dataset
# (Hugging Face `rotten_tomatoes` config).  It contains short snippets of
# professional movie reviews labelled as *positive* (1) or *negative* (0):
#
# | Split       | Reviews |
# |-------------|---------|
# | `train`     |   8 530 |
# | `test`      |   1 066 |
# | `validation`|   1 066 |
#
# The task is **binary sentiment classification**: given the text of a review,
# predict whether the critic liked the movie.  This is a classic bag-of-words
# benchmark — small enough to train in seconds on a laptop, yet rich enough to
# expose the differences between TF-IDF, transformer embeddings, and
# generative prompting.
#
# We load the dataset with `datasets.load_dataset` and convert each split to a
# `pandas.DataFrame` so we can inspect it with the tools we already know.

# %%
from datasets import load_dataset

data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

train_df = data["train"].to_pandas()
test_df = data["test"].to_pandas()
val_df = data["validation"].to_pandas()

print(f"Train : {len(train_df):,} reviews")
print(f"Test  : {len(test_df):,} reviews")
print(f"Val   : {len(val_df):,} reviews")
print()
print("Label distribution (train):")
print(train_df["label"].value_counts().rename({0: "negative", 1: "positive"}).to_string())
print()
train_df.head()

# %% [markdown]
# Each row has two columns: `text` (the review snippet) and `label`
# (0 = negative, 1 = positive).  The classes are balanced, so accuracy is a
# reasonable headline metric — but we will also report precision, recall and
# F1 to catch any asymmetric behaviour.

# %% [markdown]
# ## 2. TF-IDF: converting text to numerical features
#
# Machine-learning models operate on **numbers**, not strings.  The simplest
# way to turn a document into a numeric vector is the **bag-of-words**
# representation: count how many times each word appears.  But raw counts are
# dominated by frequent, uninformative words ("the", "and", "is").
#
# **TF-IDF** (Term Frequency × Inverse Document Frequency) down-weights words
# that appear in many documents:
#
# $$\text{tfidf}(t, d) = \text{tf}(t, d) \times \log\frac{N}{\text{df}(t)}$$
#
# - **tf(t, d)** — how often term $t$ appears in document $d$.
# - **df(t)** — in how many documents $t$ appears at all.
# - **N** — total number of documents.
#
# A word that occurs in *every* review gets an IDF near zero; a word that
# appears in only a handful gets a large IDF.  The resulting matrix is
# **sparse** — most words do not appear in most reviews — so scikit-learn
# stores it in a compressed sparse format instead of a dense array.
#
# Let us fit a `TfidfVectorizer` on the first five training reviews and look
# at the resulting feature matrix as a `DataFrame`.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

sample_texts = train_df["text"].head(5)
tfidf_demo = TfidfVectorizer()
tfidf_matrix = tfidf_demo.fit_transform(sample_texts)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_demo.get_feature_names_out(),
    index=[f"review_{i}" for i in range(5)],
)
print(f"Sparse matrix shape : {tfidf_matrix.shape}")
print(f"Non-zero entries    : {tfidf_matrix.nnz}")
tfidf_df.iloc[:, :12]

# %% [markdown]
# Each row is a review, each column a word in the vocabulary, and each cell
# the TF-IDF weight.  Notice how many zeros there are — even with only five
# documents the matrix is already mostly empty.  This is why sparse storage
# is essential for text work.

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 1 — Building a TF-IDF classifier</b>
# <ul>
#   <li>
#     Build a pipeline combining <code>TfidfVectorizer()</code> and
#     <code>LogisticRegression(max_iter=1000, random_state=42)</code>.
#   </li>
#   <li>
#     Split the training data with
#     <code>train_test_split(..., test_size=0.2, random_state=42)</code>.
#   </li>
#   <li>
#     Fit the pipeline and report accuracy on the test split.
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ## 3. Classifying reviews with TF-IDF
#
# Now let us build a **working** TF-IDF + Logistic Regression pipeline on the
# full training set and evaluate it on the held-out test split.  This is the
# baseline every later section (BERT embeddings, zero-shot prompting) must
# beat.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_text = list(train_df["text"])
y = train_df["label"].values

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

tfidf_clf = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000, random_state=42),
)
tfidf_clf.fit(X_train_text, y_train)

y_pred = tfidf_clf.predict(X_val_text)
print("=== TF-IDF + Logistic Regression (validation split) ===")
print(classification_report(y_val, y_pred, target_names=["negative", "positive"]))

# %% [markdown]
# A linear model on TF-IDF features typically reaches **~85-88 %** accuracy
# on this split — a strong baseline for such a simple representation.  The
# pipeline handles the entire flow: raw text → sparse TF-IDF matrix →
# logistic regression.  No manual feature engineering required.
#
# Can a different linear model or a tree ensemble do better?  Let us compare.

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 2 — Comparing classifiers on TF-IDF features</b>
# <ul>
#   <li>
#     Compare <code>LogisticRegression(max_iter=1000, random_state=42)</code>,
#     <code>LinearSVC(random_state=42)</code> and
#     <code>RandomForestClassifier(random_state=42)</code> on TF-IDF features
#     using 5-fold <code>cross_val_score</code> (use the full training set).
#   </li>
#   <li>
#     Report the mean ± std accuracy for each model.  Which model wins?
#   </li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ## 4. Dimensionality reduction on TF-IDF features
#
# TF-IDF vectorisation produces a **high-dimensional sparse matrix**: there is one
# column per unique token in the vocabulary, which can easily reach tens of
# thousands of dimensions even for a modest corpus. Many of those dimensions are
# noisy (rare words that appear once) or redundant (synonyms, morphological
# variants that co-occur).
#
# **Principal Component Analysis (PCA)** projects the data onto the directions
# of greatest variance, letting us compress thousands of sparse TF-IDF
# dimensions into a handful of dense, informative components. Below we fit a
# `TfidfVectorizer` on the Rotten Tomatoes training set and inspect how much variance
# each principal component captures.

# %%
from sklearn.decomposition import PCA

# Fit TfidfVectorizer on the full training set
tfidf_full = TfidfVectorizer(max_features=2000)
X_tfidf_full = tfidf_full.fit_transform(train_df["text"])
print(f"TF-IDF matrix shape: {X_tfidf_full.shape}")

# PCA on TF-IDF features
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_tfidf_full.toarray())

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, 21), pca.explained_variance_ratio_ * 100)
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance (%)")
ax.set_title("Scree plot — PCA on TF-IDF features (20 components)")
plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-success">
# <b>EXERCISE 3 — Kernel PCA for non-linear reduction</b>
# <ul>
#   <li>Apply <code>KernelPCA(kernel='rbf')</code> inside the TF-IDF pipeline.</li>
#   <li>Use <code>GridSearchCV</code> to tune <code>n_components</code> ∈ {5, 10, 20} and <code>gamma</code> ∈ {1e-3, 1e-2, 1e-1}.</li>
#   <li>Does non-linear reduction help compared to linear PCA?</li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ## 5. Beyond TF-IDF: sentence embeddings with pre-trained BERT
#
# TF-IDF represents each review as a **bag of words**: word order is discarded and
# two reviews that use different words to say the same thing look completely
# different to the classifier. Pre-trained transformer models such as **BERT**
# take the opposite approach: they read the whole sentence and map it to a
# single **dense 768-dimensional vector** (an *embedding*) that captures
# semantic meaning.
#
# We use [`sentence-transformers`](https://www.sbert.net/), a library built on
# top of Hugging Face `transformers` that fine-tunes BERT-style models to
# produce sentence-level embeddings suitable for similarity and classification
# tasks. The model
# [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
# is one of the best general-purpose sentence encoders available.
#
# **Expectation:** encoding a few thousand sentences with a 110M-parameter
# model takes roughly **30–90 seconds on CPU** (much faster on a GPU/Colab T4).
# A progress bar is shown while encoding.

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
print(f"Train embeddings shape: {train_embeddings.shape}")

# %% [markdown]
# Each row of `train_embeddings` is a 768-dimensional vector summarising one
# review. To evaluate a classifier we also need the embeddings for the test
# reviews — that is left as the next exercise.

# %% [markdown]
# <div class="alert alert-success">
# <b>EXERCISE 4 — BERT embeddings + Logistic Regression</b>
# <ul>
#   <li>Compute <code>test_embeddings</code> using the same <code>model.encode()</code> on <code>data["test"]["text"]</code>.</li>
#   <li>Train a <code>LogisticRegression(max_iter=1000, random_state=42)</code> on <code>train_embeddings</code>.</li>
#   <li>Evaluate on <code>test_embeddings</code> and compare accuracy with the TF-IDF classifier from Section 3.</li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# <div class="alert alert-success">
# <b>EXERCISE 5 — Visualizing BERT embeddings</b>
# <ul>
#   <li>Apply <code>PCA(n_components=2)</code> to the BERT embeddings.</li>
#   <li>Create a scatter plot colored by sentiment label (0 = negative, 1 = positive).</li>
#   <li>Do positive and negative reviews form separable clusters?</li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ## 6. Zero-shot classification with a generative LLM (T5)
#
# **Zero-shot classification** means classifying text *without any task-specific
# training*. Instead of fitting a classifier on labelled examples, we frame the
# task as a natural-language prompt and let a pre-trained generative model
# produce the answer directly. The model's only "knowledge" of the task comes
# from the wording of the prompt.
#
# We use **`google/flan-t5-small`** as the example model because it is small
# (~80 MB), instruction-tuned (so it follows prompts reasonably well), and runs
# on a CPU — no GPU required. Larger models (flan-t5-base, flan-t5-large, or
# decoder-only LLMs) would give better accuracy but need a GPU and are slower.
#
# > **This section is OPTIONAL.** It downloads a model from Hugging Face and
# > runs inference, which is slow on CPU (several minutes for 50 samples). It
# > is **CI-guarded** — it will be skipped when the notebook is executed under
# > `nbmake` in GitHub Actions, so it never blocks the test suite.

# %%
if not os.environ.get("CI"):
    import torch
    from transformers import pipeline as tpipeline
    from transformers.pipelines.pt_utils import KeyDataset
    from tqdm import tqdm

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running T5 on {device} (this may take a while on CPU)")

    pipe = tpipeline("text2text-generation", model="google/flan-t5-small", device=device)

    # Working example prompt — not empty string!
    prompt = "Is this movie review positive or negative? Answer with 'positive' or 'negative'. Review: "
    llm_response = data.map(lambda example: {"t5": prompt + example["text"]})

    # Use a small subset for demo (not all 1000 test samples)
    n_demo = 50
    test_subset = llm_response["test"].select(range(n_demo))
    y_pred = []
    for output in tqdm(pipe(KeyDataset(test_subset, "t5"))):
        text = output[0]["generated_text"].lower().strip()
        if "pos" in text:
            y_pred.append(1)
        elif "neg" in text:
            y_pred.append(0)
        else:
            y_pred.append(0)  # default to negative for unexpected output

    from sklearn.metrics import accuracy_score

    y_true = data["test"]["label"][:n_demo]
    print(f"T5 zero-shot accuracy (on {n_demo} samples): {accuracy_score(y_true, y_pred):.2f}")
else:
    print("Skipping T5 demo in CI (requires model download + slow inference).")

# %% [markdown]
# <div class="alert alert-success">
# <b>EXERCISE 6 — Prompt engineering for zero-shot classification</b>
# <ul>
#   <li>Try different prompt formulations. Can you improve the zero-shot accuracy?</li>
#   <li>Compare with the BERT+sklearn classifier from Exercise 4.</li>
#   <li>What are the trade-offs between TF-IDF, BERT embeddings, and zero-shot prompting?</li>
# </ul>
# </div>

# %%
# Your code here

# %% [markdown]
# ---
# ## Summary
#
# In this notebook you:
# - Loaded and explored the Rotten Tomatoes sentiment dataset.
# - Built a **TF-IDF + Logistic Regression** text classifier as a strong baseline (~85-88% accuracy).
# - Applied **PCA** to compress sparse TF-IDF features and experimented with **Kernel PCA** for non-linear reduction.
# - Extracted **dense sentence embeddings** with a pre-trained BERT model (`all-mpnet-base-v2`).
# - Compared supervised classification against **zero-shot T5 prompting**.
#
# **Key take-away:** pre-trained LLMs provide powerful representations, but even
# simple TF-IDF baselines can be surprisingly competitive on binary sentiment tasks.
# The choice between TF-IDF, BERT embeddings, and zero-shot prompting depends on
# your constraints: labelled data availability, compute budget, and latency requirements.
