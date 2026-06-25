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

# %% [markdown] id="_-Zkgb3dAA3c"
# # An exercise in text classification: Guessing sentiment from movie criticism
#
# We are going to work on a database of movie reviews. Each review has been annotated as positive (1) or negative (0).
#
# To work in this case we will use a T4 type of virtual computer. Select it on the upper right side of colab, where it states "Connect", go to "Change Runtime type"
#

# %% [markdown] id="CJ_NWfasAeFZ"
# %% [markdown]
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Load and explore a text classification dataset (Rotten Tomatoes sentiment).
# 2. Represent text as TF-IDF vectors and classify with Logistic Regression.
# 3. Apply dimensionality reduction (PCA, Kernel PCA) to sparse text features.
# 4. Extract dense sentence embeddings with a pre-trained transformer (BERT family).
# 5. Use a generative LLM (T5) for zero-shot classification via prompting.

# ## 1. Download and import the dataset.

 # %% id="qwKHxrqb4FN5"
 
# %% id="HheXxTsD46Po"
from datasets import load_dataset
import pandas as pd

# %% id="3QLS2eTq4GqR"
data = load_dataset("rotten_tomatoes")
train_data = pd.DataFrame(data['train'])

# %% [markdown] id="Vt7Rs8-MAqfG"
# ## 2. Explore the data

# %% id="npuIvbCz4YuN"

# %% [markdown] id="igMnAdXkA0EB"
# # 3. Converting text into numerical features
#
# We need to convert the text into a representation amenable to a machine learning
# algorithm. For this we will use the `TfidfVectorizer`.
#
# For this we will use the TfidfVectorizer. Test how this works on the first 5 texts

# %% id="oGAETA08BApH"
from sklearn.feature_extraction.text import TfidfVectorizer

# %% id="xoUjGxJgBr6P"
train_data.iloc[:5].text

# %% id="dMvkunpbBBhi"
vectorizer = TfidfVectorizer()
pd.DataFrame(
    vectorizer.fit_transform(train_data.iloc[:5].text).todense(),
    columns=vectorizer.get_feature_names_out()
)

# %% [markdown] id="VeZ15xNaB6T9"
# ## 4. Classify the reviews
#
# Use the Logistic regression classifier and the TfidfVectorizer to classify the *reviews*.

# %% id="5HbtEZnUAxNX"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    LearningCurveDisplay, StratifiedKFold, train_test_split
)

# %% id="2DoA3t_OB3vv"

# %% [markdown] id="_KLPXcyyneGL"
# ## 4.1 Dimensionality reduction on TF-IDF features
#
# TF-IDF produces a very high-dimensional sparse matrix (one dimension per unique token).
# Many of those dimensions are noisy or redundant. Applying PCA or Kernel PCA can
# sometimes help by projecting onto the most informative directions.
#
# Apply PCA and then Kernel PCA inside the pipeline (after TF-IDF, before the
# classifier). Use at most 20 components. Use the learning curve to choose the
# number of components. Does compression help or hurt accuracy?
#
#

# %% id="oPArG-cwnsu_"
from sklearn.decomposition import PCA, KernelPCA

# %% id="k-Wf_M_ins0k"

# %% [markdown] id="zOYW8dJkCqmC"
# ## 5. Large language models !
#
# Now we will use a large language model [bert](https://en.wikipedia.org/wiki/BERT_(language_model))
# as a feature extractor

# %% id="hcztM8dy5oNW"
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)

# %% [markdown] id="BJecahf9C0tU"
# Now that we have extracted the features to a new variable `train_embeddings` try the classifier again. Use these features as input for the classifier.
#

# %% id="WTCXeTewHGRr"

# %% id="68bG22kqHFui"

# %% [markdown] id="wX4SiSNCEE3j"
# ## Just as an example. LLM Prompting !
# Now let's just ask an LLM for this, specifically [T5](https://en.wikipedia.org/wiki/T5_(language_model))
# We will use the test data for convenience as it has only around 1000 records

# %% id="nDuW0ExY-dIx"
from transformers import pipeline as tpipeline
from transformers.pipelines.pt_utils import KeyDataset
import numpy as np
from tqdm import tqdm

# %% id="_iKtCUkE9-gM"
# Load our model, if you specify device="cuda:0" it will
# consume the free computing from google very fast
# if not change device="cpu"
pipe = tpipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device="cuda:0"
)

# %% id="uYDMbfnkGiE4"
# Store the data in a dataframe

data_df = pd.DataFrame(data["test"])

# %% id="I-ryGvFv-XTE"
# Set up the prompt that we will ask to the T5 llm. How are you going to ask ?
prompt = "" # Ask here if the text is positive or negative, try different formulations
llm_response = data.map(lambda example: {"t5": prompt + example['text']})

# %% id="GYCMdj01-sku"
# Run the prompt for every record and store the result

y_pred = []
for output in tqdm(pipe(KeyDataset(llm_response["test"], "t5"))):
    text = output[0]["generated_text"]
    y_pred.append(0 if text == "negative" else 1)

# %% id="EMm_mpN1_GKO"
# Convert the result to pandas
y_pred = pd.Series(y_pred)

# %% id="lVIT_W3IDHav"
# Compute the accuracy
from sklearn.metrics import accuracy_score

print(f"Accuracy {accuracy_score(data_df.label, y_pred):.2f}")

# %% id="kwXRrevmDr8-"

# %% [markdown]
# ---
# ## Summary
#
# In this notebook you:
# - Loaded and explored the Rotten Tomatoes sentiment dataset.
# - Built a TF-IDF + Logistic Regression text classifier.
# - Experimented with PCA and Kernel PCA to compress TF-IDF features.
# - Used BERT-family sentence embeddings as richer features.
# - Compared supervised classification against a zero-shot T5 prompt.
#
# **Key take-away:** pre-trained LLMs provide powerful representations but even
# simple TF-IDF baselines can be surprisingly competitive on binary sentiment tasks.
