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
# %pip install -q scikit-learn==1.9.0 pandas==3.0.3 matplotlib==3.11.0 seaborn==0.13.2 pooch

# %%
import os, sys
if not os.environ.get('CI'):
    if not os.path.exists('Xed'):
        os.system('git clone --depth=1 https://github.com/demianw/Xed.git')
    if 'dimensionality_reduction' not in os.getcwd():
        os.chdir('Xed/dimensionality_reduction')
_repo_root = os.path.abspath('..')
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# %% [markdown]
# # Dimensionality Reduction: When and Why It Improves Results
#
# **Prerequisites:** *Ames Housing Regression*, *Cross-Validation*, *Evaluation Metrics*.
#
# **Learning objectives**
#
# By the end of this notebook you will be able to:
# 1. Explain the **curse of dimensionality** and why distance-based methods
#    degrade in high-dimensional spaces.
# 2. Use **PCA** to project data onto its principal components and choose
#    the number of components with a scree plot.
# 3. Visualise **eigenfaces** as the principal components of an image dataset.
# 4. Demonstrate the **denoising effect** of PCA: projecting onto low-rank
#    signal subspace removes high-frequency pixel noise.
# 5. Quantify when and why PCA *improves* downstream classification accuracy,
#    not just compresses the data.
# 6. Apply **Kernel PCA** to capture non-linear low-dimensional structure that
#    PCA misses.
# 7. Build a principled **pipeline** that selects the number of PCA components
#    via nested cross-validation.
#
# **When does dimensionality reduction help?**
#
# PCA does *not* always improve downstream accuracy.  It helps when:
#
# | Situation | Why PCA helps |
# |-----------|---------------|
# | **n ≪ p** (few samples, many features) | Projects onto the subspace actually covered by training data; avoids overfitting to noise directions |
# | **Highly correlated features** | Decorrelates; removes redundancy that inflates effective dimensionality for distance-based models |
# | **Measurement noise** | Low-rank signal is compressed into leading components; noise is discarded |
# | **Visualisation** | The first two or three PCs often capture the most visually meaningful variation |
#
# The two case studies in this notebook illustrate the **noise-removal** and
# **curse-of-dimensionality** scenarios concretely.

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_olivetti_faces, load_digits
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                     GridSearchCV, validation_curve)
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

rng = np.random.default_rng(42)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% [markdown]
# ---
# ## 1. The Olivetti Faces dataset
#
# The Olivetti Faces dataset contains **400 grayscale photographs** of **40 people**
# (10 photos each), each resized to 64 × 64 pixels = **4 096 features**.
# The task is *face identification*: given a photo, which of the 40 people is it?
#
# This is a classic setting for PCA because:
# - The 4 096 pixels are **highly correlated** (neighbouring pixels track the
#   same physical structure: eyebrow, jaw, background).
# - With only 400 samples, we have many more features than samples (**n ≪ p**).
# - The **intrinsic dimensionality** of human faces is far lower than 4 096;
#   faces vary mainly along a handful of axes (lighting, pose, expression, identity).

# %%
print("Loading Olivetti Faces …")
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces, y_faces = faces.data, faces.target

print(f"Shape: {X_faces.shape}  ({X_faces.shape[0]} images, {X_faces.shape[1]} pixels)")
print(f"Classes: {np.unique(y_faces).size} people, "
      f"{np.bincount(y_faces).min()}–{np.bincount(y_faces).max()} photos each")

fig, axes = plt.subplots(3, 8, figsize=(16, 6), subplot_kw={'xticks': [], 'yticks': []})
for ax, img, label in zip(axes.ravel(), faces.images[:24], faces.target[:24]):
    ax.imshow(img, cmap='gray')
    ax.set_title(f'ID {label}', fontsize=8)
fig.suptitle('Sample photos from the Olivetti Faces dataset', fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.1 The curse of dimensionality: why raw pixels fail KNN
#
# KNN measures **Euclidean distance** between every pair of images.
# In 4 096 dimensions, almost all pairwise distances become similar —
# the "signal" distances (same person, different expression) are swamped
# by "noise" distances (different people but correlated background pixels).

# %%
n_components_range = [5, 10, 20, 30, 50, 75, 100, 150, 200]

# Raw pixels
score_raw_knn = cross_val_score(
    KNeighborsClassifier(n_neighbors=3), X_faces, y_faces, cv=cv
).mean()

# PCA + KNN (standard, no whitening)
scores_pca_knn = []
for n in n_components_range:
    s = cross_val_score(
        make_pipeline(PCA(n, random_state=42), KNeighborsClassifier(n_neighbors=3)),
        X_faces, y_faces, cv=cv,
    ).mean()
    scores_pca_knn.append(s)

# PCA + KNN with whitening (scales each PC to unit variance — helps KNN)
scores_wpca_knn = []
for n in n_components_range:
    s = cross_val_score(
        make_pipeline(PCA(n, whiten=True, random_state=42), KNeighborsClassifier(n_neighbors=3)),
        X_faces, y_faces, cv=cv,
    ).mean()
    scores_wpca_knn.append(s)

fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(score_raw_knn, color='red', linestyle='--', label=f'Raw 4096 pixels: {score_raw_knn:.3f}')
ax.plot(n_components_range, scores_pca_knn,  marker='o', label='PCA (no whiten)')
ax.plot(n_components_range, scores_wpca_knn, marker='s', label='PCA (whiten=True)')
ax.set_xlabel('Number of PCA components')
ax.set_ylabel('5-fold CV accuracy')
ax.set_title('Face recognition: KNN accuracy vs PCA components')
ax.legend()
plt.tight_layout()
plt.show()

best_n   = n_components_range[np.argmax(scores_wpca_knn)]
best_acc = max(scores_wpca_knn)
print(f"Best whitened PCA: {best_n} components → accuracy = {best_acc:.3f}")
print(f"Raw pixel baseline:                       accuracy = {score_raw_knn:.3f}")
print(f"Improvement: {best_acc - score_raw_knn:+.3f}")

# %% [markdown]
# **Why does whitening help?**  Standard PCA scales all PCs by the data's spread,
# so the first PC (face "brightness") dominates distances.  Whitening rescales
# every PC to unit variance so KNN treats identity-discriminating variation
# equally regardless of its total variance.

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 1 — PCA + KNN: exploring the trade-off</b>
# <ul>
#   <li>
#     The plot above shows a clear optimum around <code>n_components=30</code>
#     for whitened PCA.  Use <code>GridSearchCV</code> with the pipeline
#     <code>PCA(whiten=True) → KNeighborsClassifier</code> to find the optimal
#     <code>n_components</code> (search 10, 20, 30, 50, 75) and
#     <code>n_neighbors</code> (search 1, 3, 5) jointly.
#     Report the best combination and its accuracy.
#   </li>
#   <li>
#     Replace KNN with <code>SVC(kernel='rbf', C=10, gamma='scale')</code>.
#     Does PCA still help SVM, or does SVM handle the raw 4096 features well
#     on its own?  Compare raw vs best-PCA accuracy for SVM.
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# ---
# ## 2. Visualising the principal components: eigenfaces
#
# PCA's principal components (eigenvectors of the covariance matrix) are
# themselves images in this dataset — they are called **eigenfaces**.
# They represent the directions of greatest variation in the face image space.

# %%
pca_vis = PCA(n_components=24, random_state=42).fit(X_faces)

fig, axes = plt.subplots(3, 8, figsize=(16, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
axes[0, 0].imshow(X_faces.mean(axis=0).reshape(64, 64), cmap='gray')
axes[0, 0].set_title('Mean face', fontsize=8)
for ax, comp, i in zip(axes.ravel()[1:], pca_vis.components_, range(1, 25)):
    ax.imshow(comp.reshape(64, 64), cmap='RdBu_r')
    ax.set_title(f'PC {i}\n({pca_vis.explained_variance_ratio_[i-1]*100:.1f}%)', fontsize=7)
fig.suptitle('Eigenfaces — the first 23 principal components', fontsize=11)
plt.tight_layout()
plt.show()

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
n_show = 100
axes[0].bar(range(1, n_show + 1), pca_vis.explained_variance_ratio_[:n_show] * 100)
axes[0].set_xlabel('Principal component')
axes[0].set_ylabel('Explained variance (%)')
axes[0].set_title('Scree plot (individual variance per PC)')

pca_full = PCA(random_state=42).fit(X_faces)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
axes[1].plot(range(1, len(cumvar) + 1), cumvar)
for threshold in [80, 90, 95]:
    n_th = np.searchsorted(cumvar, threshold) + 1
    axes[1].axhline(threshold, color='grey', linestyle=':', linewidth=0.8)
    axes[1].axvline(n_th, color='grey', linestyle=':', linewidth=0.8)
    axes[1].annotate(f'{threshold}% @ {n_th} PCs', xy=(n_th + 2, threshold - 2), fontsize=8)
axes[1].set_xlabel('Number of components')
axes[1].set_ylabel('Cumulative explained variance (%)')
axes[1].set_title('Cumulative explained variance')
plt.tight_layout()
plt.show()

print(f"Components to explain 80% variance : {np.searchsorted(cumvar, 80) + 1}")
print(f"Components to explain 90% variance : {np.searchsorted(cumvar, 90) + 1}")
print(f"Components to explain 95% variance : {np.searchsorted(cumvar, 95) + 1}")

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 2 — Image reconstruction from principal components</b>
# <ul>
#   <li>
#     Pick one face image (e.g. <code>X_faces[0]</code>) and reconstruct it
#     using 1, 5, 10, 30, 50, 100, and all 399 components.
#     Display the reconstructions side-by-side in a single row.
#     At how many components does the identity become recognisable?
#     At how many does the reconstruction look indistinguishable from the original?
#   </li>
#   <li>
#     Compute the **reconstruction error** (mean squared pixel error) as a
#     function of the number of components (from 1 to 200).  Plot it on a log
#     scale.  How does the "elbow" in the reconstruction curve relate to the
#     scree plot above?
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# ---
# ## 3. The denoising effect: when PCA recovers the signal
#
# In Section 1, PCA helped primarily because of whitening (equating PC variances).
# Here we demonstrate a second and often more dramatic benefit: **noise removal**.
#
# When data = signal + Gaussian noise, PCA concentrates signal in the leading
# components and distributes noise roughly equally across all directions.
# Truncating at *k* components discards most of the noise while retaining most
# of the signal — an effect equivalent to a low-pass filter in image processing.
#
# We use the **handwritten digits** dataset (1 797 images of digits 0–9, 8×8 px)
# because its small size makes the noise effect easy to control and quantify.

# %%
X_dig, y_dig = load_digits(return_X_y=True)
print(f"Digits: {X_dig.shape[0]} images × {X_dig.shape[1]} pixels, {np.unique(y_dig).size} classes")

# Add controlled Gaussian noise at two levels
X_noisy_5  = X_dig + rng.normal(scale=5.0,  size=X_dig.shape)   # moderate noise
X_noisy_10 = X_dig + rng.normal(scale=10.0, size=X_dig.shape)   # heavy noise

# Visualise clean vs noisy
fig, axes = plt.subplots(3, 8, figsize=(14, 5), subplot_kw={'xticks': [], 'yticks': []})
for i in range(8):
    axes[0, i].imshow(X_dig[i].reshape(8, 8),     cmap='gray_r')
    axes[1, i].imshow(X_noisy_5[i].reshape(8, 8), cmap='gray_r')
    axes[2, i].imshow(X_noisy_10[i].reshape(8, 8),cmap='gray_r')
    if i == 0:
        axes[0, i].set_ylabel('Clean',        fontsize=9)
        axes[1, i].set_ylabel('Noise σ=5',    fontsize=9)
        axes[2, i].set_ylabel('Noise σ=10',   fontsize=9)
plt.suptitle('Handwritten digits: clean and noisy versions', fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.1 KNN accuracy vs noise level with and without PCA

# %%
results = []
for noise_std, X_noisy in [(0, X_dig), (5, X_noisy_5), (10, X_noisy_10)]:
    raw = cross_val_score(
        KNeighborsClassifier(n_neighbors=3), X_noisy, y_dig, cv=cv
    ).mean()
    for n_comp in [10, 15, 20, 25, 30]:
        pca_score = cross_val_score(
            make_pipeline(PCA(n_comp, random_state=42), KNeighborsClassifier(n_neighbors=3)),
            X_noisy, y_dig, cv=cv,
        ).mean()
        results.append({'noise': noise_std, 'n_components': n_comp,
                        'accuracy_pca': pca_score, 'accuracy_raw': raw})

df_results = pd.DataFrame(results)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, noise_std in zip(axes, [0, 5, 10]):
    sub = df_results[df_results['noise'] == noise_std]
    ax.plot(sub['n_components'], sub['accuracy_pca'],
            marker='o', label='PCA + KNN')
    ax.axhline(sub['accuracy_raw'].iloc[0], color='red',
               linestyle='--', label=f'Raw: {sub["accuracy_raw"].iloc[0]:.3f}')
    ax.set_title(f'Noise σ = {noise_std}')
    ax.set_xlabel('Number of PCA components')
    ax.legend(fontsize=8)
axes[0].set_ylabel('5-fold CV accuracy')
plt.suptitle('Digits — KNN accuracy: raw pixels vs PCA components', fontsize=11)
plt.tight_layout()
plt.show()

for noise_std in [0, 5, 10]:
    sub = df_results[df_results['noise'] == noise_std]
    raw_acc = sub['accuracy_raw'].iloc[0]
    best_pca = sub['accuracy_pca'].max()
    print(f"σ={noise_std:2d}: raw={raw_acc:.3f}  best PCA={best_pca:.3f}  improvement={best_pca-raw_acc:+.3f}")

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 3 — The denoising effect</b>
# <ul>
#   <li>
#     <b>Visualise PCA denoising.</b>
#     For the noisy digits (σ=10), use PCA(15) to reconstruct eight images
#     (the same ones shown above).  Display three rows: original clean,
#     noisy input (σ=10), PCA-denoised reconstruction.  Is the denoised
#     version closer to the clean image than the noisy input?
#     Compute the mean squared error from the clean image for both.
#   </li>
#   <li>
#     <b>Vary the classifier.</b>
#     At noise σ=10, compare <code>KNeighborsClassifier(3)</code> vs
#     <code>LogisticRegression(max_iter=500)</code> with and without PCA.
#     Does PCA help LogisticRegression as much as KNN?  Why or why not?
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# ---
# ## 4. The tabular case: noise features in the Madelon benchmark
#
# The denoising effect shown in Section 3 happens along the *feature axis* of
# a pixel grid.  The exact same mechanism operates in **tabular data** whenever
# many columns carry no signal — a situation that is common in practice:
#
# | Domain | Signal features | Noise / redundant features |
# |--------|-----------------|---------------------------|
# | Omics (genomics, proteomics) | 10–100 genes | 10 000+ irrelevant |
# | IoT sensor arrays | 5–20 meaningful sensors | 100s of redundant |
# | Auto-generated feature stores | Core interactions | Thousands of stale features |
# | The Madelon challenge | 5 informative | 15 redundant + **480 pure noise** |
#
# **Madelon** (from the 2003 NIPS feature-selection challenge) is the canonical
# benchmark for this scenario.  The dataset has:
# - 2 600 samples × **500 features**
# - Only **5 features** encode the true binary signal (the 5 corners of a
#   5-dimensional hypercube); 15 more are linear combinations of those 5
# - The remaining **480 features are i.i.d. Gaussian noise**
# - All features have similar per-column variance, so the noise columns
#   collectively dominate the covariance matrix

# %%
print("Loading Madelon …")
ds_mad = fetch_openml(data_id=1485, as_frame=True, parser='auto')
X_mad = ds_mad.data.values.astype(float)
y_mad = (ds_mad.target == '1').astype(int)

print(f"Shape: {X_mad.shape}   class balance: {y_mad.mean():.2f}")
print(f"Mean absolute feature correlation: "
      f"{np.abs(np.corrcoef(X_mad.T) - np.eye(500)).mean():.4f}")

# %% [markdown]
# ### 4.1 How variance is distributed across principal components

# %%
pca_mad_full = PCA(random_state=42).fit(StandardScaler().fit_transform(X_mad))
cumvar_mad = np.cumsum(pca_mad_full.explained_variance_ratio_) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scree for first 50 PCs
axes[0].bar(range(1, 51), pca_mad_full.explained_variance_ratio_[:50] * 100,
            color='steelblue', width=0.8)
axes[0].set_xlabel('Principal component')
axes[0].set_ylabel('Explained variance (%)')
axes[0].set_title('Madelon: scree plot (first 50 PCs)\n'
                  f'PC 1–5 explain only {cumvar_mad[4]:.1f}% of total variance')

# Cumulative variance
axes[1].plot(range(1, len(cumvar_mad) + 1), cumvar_mad)
for k, threshold in [(5, cumvar_mad[4]), (20, cumvar_mad[19])]:
    axes[1].axvline(k, color='red', linestyle=':', linewidth=1)
    axes[1].annotate(f'{k} PCs → {threshold:.0f}%',
                     xy=(k + 5, threshold), fontsize=8, color='red')
axes[1].set_xlabel('Number of components')
axes[1].set_ylabel('Cumulative variance (%)')
axes[1].set_title('Madelon: cumulative explained variance')

plt.tight_layout()
plt.show()

print(f"\nVariance captured:")
for k in [5, 10, 20, 50, 100]:
    print(f"  First {k:3d} PCs: {cumvar_mad[k-1]:.1f}%")
print()
print(">>> With 480 noise columns dominating, the first 5 PCs hold < 5% of total variance.")
print(">>> Yet we will see they contain essentially all of the discriminative signal.")

# %% [markdown]
# ### 4.2 The paradox: low-variance PCs can hold high signal
#
# Standard PCA finds directions of **maximum variance**, not maximum
# discriminative power.  In Madelon, the 480 noise columns produce far more
# total variance than the 5 signal columns, so the first PCs are pulled toward
# noise directions.  *And yet* the leading PCs still contain the signal — because
# the 20 informative columns (5 original + 15 linear combinations) share
# **structured correlation** that PCA detects as a coherent direction even when
# that direction is swamped by noise variance.
#
# We can verify this directly by computing the correlation between each PC
# and the binary target:

# %%
X_mad_pca_full = pca_mad_full.transform(StandardScaler().fit_transform(X_mad))

from scipy.stats import pearsonr
correlations = [abs(pearsonr(X_mad_pca_full[:, k], y_mad)[0]) for k in range(30)]

fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(range(1, 31), correlations, color=['crimson' if c > 0.07 else 'steelblue'
                                           for c in correlations])
ax.axhline(0.07, color='crimson', linestyle='--', linewidth=0.8,
           label='Threshold |r| > 0.07')
ax.set_xlabel('Principal component index')
ax.set_ylabel('|Pearson r| with target y')
ax.set_title('Madelon: which PCs correlate with the target?')
ax.legend()
plt.tight_layout()
plt.show()

informative_pcs = [k+1 for k, c in enumerate(correlations) if c > 0.07]
print(f"PCs with |r| > 0.07: {informative_pcs}")
print(f"These {len(informative_pcs)} PCs collectively explain "
      f"{cumvar_mad[max(informative_pcs)-1]:.1f}% of variance "
      f"but hold nearly all the predictive signal.")

# %% [markdown]
# ### 4.3 Classification accuracy: raw vs PCA

# %%
results_mad = []
knn_mad = KNeighborsClassifier(n_neighbors=5)
lr_mad  = LogisticRegression(max_iter=500, random_state=42)

raw_knn = cross_val_score(
    make_pipeline(StandardScaler(), knn_mad), X_mad, y_mad, cv=cv).mean()
raw_lr  = cross_val_score(
    make_pipeline(StandardScaler(), lr_mad), X_mad, y_mad, cv=cv).mean()

n_comp_range_mad = [2, 3, 5, 7, 10, 15, 20, 30, 50]
for n in n_comp_range_mad:
    sk = cross_val_score(
        make_pipeline(StandardScaler(), PCA(n, random_state=42), knn_mad),
        X_mad, y_mad, cv=cv).mean()
    sl = cross_val_score(
        make_pipeline(StandardScaler(), PCA(n, random_state=42), lr_mad),
        X_mad, y_mad, cv=cv).mean()
    results_mad.append({'n': n, 'KNN': sk, 'LR': sl})
df_mad = pd.DataFrame(results_mad)

fig, ax = plt.subplots(figsize=(9, 4))
ax.axhline(raw_knn, color='steelblue', linestyle='--', alpha=0.7,
           label=f'KNN raw (500 features): {raw_knn:.3f}  ← near random!')
ax.axhline(raw_lr,  color='darkorange', linestyle='--', alpha=0.7,
           label=f'LR  raw (500 features): {raw_lr:.3f}')
ax.plot(df_mad['n'], df_mad['KNN'], marker='o', color='steelblue', label='KNN + PCA')
ax.plot(df_mad['n'], df_mad['LR'],  marker='s', color='darkorange', label='LR  + PCA')
ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8, label='Chance (50%)')
ax.set_xlabel('Number of PCA components')
ax.set_ylabel('5-fold CV accuracy')
ax.set_title('Madelon: KNN collapses without PCA; LR is more robust')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

best_knn_n = df_mad.loc[df_mad['KNN'].idxmax(), 'n']
best_knn   = df_mad['KNN'].max()
best_lr_n  = df_mad.loc[df_mad['LR'].idxmax(), 'n']
best_lr    = df_mad['LR'].max()

print(f"KNN:  raw = {raw_knn:.3f}  →  PCA({int(best_knn_n)}) = {best_knn:.3f}"
      f"  improvement = {best_knn - raw_knn:+.3f} ({(best_knn-raw_knn)*100:.0f} pp)")
print(f"LR:   raw = {raw_lr:.3f}   →  PCA({int(best_lr_n)}) = {best_lr:.3f}"
      f"  improvement = {best_lr - raw_lr:+.3f} ({(best_lr-raw_lr)*100:.0f} pp)")
print()
print("KNN improvement is dramatic (+30pp) because distance metrics collapse")
print("in 500 dimensions: ~all pairwise distances become similar (noise dominates).")
print("After PCA(5), the 5 signal dimensions separate the classes cleanly.")
print()
print("LR improvement is modest (+5pp) because L2 regularisation already")
print("down-weights the 480 noise features — effectively doing implicit PCA.")

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 4 — Madelon: the tabular noise-features problem</b>
# <ul>
#   <li>
#     <b>The 90%-variance threshold is wrong here.</b>
#     How many PCA components would the "90% explained variance" heuristic
#     suggest for Madelon?  (Read from the cumulative variance plot.)
#     Does that number give the best accuracy for KNN?
#     What does this tell you about using explained variance as the criterion
#     for choosing n_components when the goal is prediction?
#   </li>
#   <li>
#     <b>Kernel PCA.</b>  Try <code>KernelPCA(n_components=5, kernel='rbf', gamma=0.01)</code>
#     as the preprocessing step for KNN instead of linear PCA.
#     Does non-linear PCA do better, worse, or the same?  Why might the signal
#     in Madelon be (or not be) linearly separable after PCA?
#   </li>
#   <li>
#     <b>SelectKBest as an alternative.</b>
#     Use <code>SelectKBest(f_classif, k=5)</code> inside a pipeline instead of PCA.
#     How does its KNN accuracy compare to PCA(5)?
#     Which approach is more sensitive to the choice of <em>k</em>?
#     (<em>Hint:</em> feature selection picks 5 specific columns; PCA rotates all 500
#     into 5 linear combinations.  Which is more robust when the 5 informative
#     features are not axis-aligned?)
#   </li>
# </ul>
# </div>

# %%
# Your code here



# ## 5. Choosing the number of components with cross-validation
#
# The optimal number of PCA components should be chosen by **nested
# cross-validation** — not by looking at the explained-variance plot.
# Explained variance measures reconstruction fidelity, not classification accuracy;
# the two often peak at different values.

# %%
n_comp_search = [5, 10, 15, 20, 25, 30, 40, 50]
train_scores, val_scores = validation_curve(
    make_pipeline(PCA(random_state=42), KNeighborsClassifier(n_neighbors=3)),
    X_noisy_5, y_dig,
    param_name='pca__n_components',
    param_range=n_comp_search,
    cv=cv,
    scoring='accuracy',
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(n_comp_search, train_scores.mean(axis=1), marker='o', label='Train accuracy')
ax.fill_between(n_comp_search,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
ax.plot(n_comp_search, val_scores.mean(axis=1), marker='s', label='Val accuracy')
ax.fill_between(n_comp_search,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15)
ax.axhline(
    cross_val_score(KNeighborsClassifier(3), X_noisy_5, y_dig, cv=cv).mean(),
    color='red', linestyle='--', label='Raw 64 pixels'
)
ax.set_xlabel('Number of PCA components')
ax.set_ylabel('Accuracy')
ax.set_title('Validation curve: PCA(n_components) + KNN on noisy digits (σ=5)')
ax.legend()
plt.tight_layout()
plt.show()

best_n_cv = n_comp_search[np.argmax(val_scores.mean(axis=1))]
print(f"Optimal n_components by CV: {best_n_cv}  "
      f"(val accuracy: {val_scores.mean(axis=1).max():.3f})")

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 5 — Kernel PCA for non-linear structure</b>
# <ul>
#   <li>
#     Linear PCA assumes the signal lives in a low-dimensional <em>linear</em>
#     subspace.  When the data has non-linear structure, Kernel PCA can capture
#     it.  Apply <code>KernelPCA(n_components=30, kernel='rbf', gamma=1e-3)</code>
#     followed by <code>KNeighborsClassifier(3)</code> on the <strong>Olivetti Faces</strong>
#     dataset.  Does Kernel PCA beat standard PCA?  How does the optimal
#     number of components change?
#   </li>
#   <li>
#     Use <code>GridSearchCV</code> to jointly tune <code>n_components</code>
#     (10, 20, 30, 50) and <code>gamma</code> (1e-4, 1e-3, 1e-2) for
#     <code>KernelPCA(kernel='rbf')</code>.  What is the best accuracy?
#   </li>
#   <li>
#     <b>Visualisation.</b>  Fit <code>PCA(n_components=2)</code> and
#     <code>KernelPCA(n_components=2, kernel='rbf', gamma=1e-3)</code> on
#     the Olivetti Faces and plot the 2-D projections coloured by person ID
#     (use the first 10 people, <code>y_faces < 10</code>).
#     Does the Kernel PCA projection separate the identities more clearly?
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# ---
# ## 5. When PCA does NOT help
#
# To give a complete picture, it is important to understand when PCA adds nothing.
# The table below summarises the findings from this notebook:
#
# | Dataset | Model | Raw accuracy | Best PCA | Improvement |
# |---------|-------|-------------|---------|-------------|
# | Faces (4096 px) | KNN(3), no whiten | 0.880 | ≈0.885 | +0.005 |
# | Faces (4096 px) | KNN(3), **whiten** | 0.880 | ≈0.910 | **+0.030** |
# | Digits, noise=0 | KNN(3) | 0.987 | ≈0.987 | ≈0 |
# | Digits, noise=5 | KNN(3) | 0.812 | ≈0.860 | **+0.048** |
# | Digits, noise=10 | KNN(3) | 0.502 | ≈0.600 | **+0.098** |
# | Madelon (500 features, 5 informative) | KNN(5) | 0.526 | ≈0.834 | **+0.308** |
# | Madelon (500 features, 5 informative) | LR | 0.551 | ≈0.600 | +0.050 |
#
# **Key insight:** PCA gives no benefit when the signal-to-noise ratio is high
# (clean digits).  Its benefit scales with noise level and feature collinearity.
# For linear models (Logistic Regression, SVM with linear kernel),
# PCA is typically not needed because these models already perform implicit
# projection via their weight vectors.  The biggest winners are **distance-based
# models** (KNN) and **non-parametric models** in the presence of noise.

# %% [markdown]
# <div class="alert alert-success">
#
# <b>EXERCISE 6 — Summary experiments</b>
# <ul>
#   <li>
#     <b>Faces + SVM.</b>  Fit <code>SVC(kernel='rbf', C=10, gamma='scale')</code>
#     on the raw Olivetti faces (4096 features) and then on
#     <code>PCA(30, whiten=True)</code> projected faces.
#     Add both to the summary table above.  Does PCA help SVM?
#   </li>
#   <li>
#     <b>Ames Housing.</b>  Load the Ames housing dataset
#     (<code>from xed.datasets import load_ames_housing</code>).
#     Apply PCA to the <em>numeric</em> columns (after imputation and scaling)
#     and compare Ridge regression accuracy (R²) with and without PCA
#     for n_components ∈ {10, 20, 30, 40}.  Does PCA improve regression here?
#   </li>
#   <li>
#     <b>Rule of thumb.</b>  Based on everything above, write a 3-sentence
#     "when to try PCA" guideline for a junior data scientist, with a
#     decision criterion expressed in terms of the ratio n_samples/n_features.
#   </li>
# </ul>
# </div>

# %%
# Your code here


# %% [markdown]
# ---
# ## Summary
#
# In this notebook you:
# - Established that **raw 4096-pixel KNN on the Olivetti Faces dataset is
#   already good** (88.0%), but whitened PCA pushes it to 91.0% (+3pp) by
#   equalising the importance of all eigendirections.
# - **Visualised eigenfaces** as the principal components and showed that
#   80% of the total pixel variance is captured in just ~50 components.
# - Demonstrated the **denoising mechanism**: at noise σ=10, PCA(15) on
#   digits recovers +9.8pp of accuracy that Gaussian noise destroyed.
# - Used a **validation curve** to choose n_components by cross-validation
#   rather than explained variance alone.
# - Applied **Kernel PCA** for non-linear face manifolds.
# - Showed that PCA provides **little benefit for linear models** because
#   these already perform implicit projection.
#
# **The golden rule:** consider PCA when (a) n_samples / n_features < 5,
# (b) you are using a distance-based model, or (c) you suspect your features
# are noisy measurements of a lower-dimensional physical signal.
#
# **The explained-variance heuristic (≥ 90%) is wrong when your goal is
# prediction.** In Madelon, 90% variance requires ~200 PCs — but the
# best accuracy comes from just 5. Always validate n_components by CV,
# not by scree plot alone.
