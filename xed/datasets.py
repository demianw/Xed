"""
Dataset loading utilities for the Xed Cardiff Introductory Data Science course.

All remote datasets are downloaded on first call, cached in
``~/.cache/xed_datasets/`` (overridable with the ``XED_CACHE_DIR``
environment variable), and verified against a SHA-256 checksum before
being returned as a DataFrame.

Download mechanism
------------------
*pooch* (https://www.fatiando.org/pooch/) is used when available: it adds
progress bars, automatic retries, and structured cache management.  If pooch
is not installed, the module falls back to ``urllib.request`` with a manual
SHA-256 check so that the notebooks work in any environment.

Adding a new dataset
--------------------
1. Add an entry to ``_REGISTRY`` with the public URL and the SHA-256 digest.
2. Write a public ``load_<name>()`` function that calls ``_fetch()`` and
   returns a properly-formatted ``pd.DataFrame``.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(
    os.environ.get("XED_CACHE_DIR", Path.home() / ".cache" / "xed_datasets")
)

# ---------------------------------------------------------------------------
# Registry of remote datasets
# ---------------------------------------------------------------------------
# Each entry: name -> (url, sha256_hex, human-readable citation)

_REGISTRY: dict[str, tuple[str, str, str]] = {
    # ------------------------------------------------------------------
    # Titanic passenger dataset (Kaggle/ML-course format)
    # 891 rows × 11 features + PassengerId index
    # ------------------------------------------------------------------
    "titanic.csv": (
        "https://raw.githubusercontent.com/datasciencedojo/datasets"
        "/master/titanic.csv",
        "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7",
        "Titanic passenger data — datasciencedojo/datasets on GitHub. "
        "Originally compiled from the British Board of Trade inquiry (1912).",
    ),
    # ------------------------------------------------------------------
    # Ames Iowa Housing dataset (original DeCode format, tab-separated)
    # 2930 rows × 82 columns; column names match the course notebooks
    # ------------------------------------------------------------------
    "AmesHousing.txt": (
        "http://jse.amstat.org/v19n3/decock/AmesHousing.txt",
        "6cfe6cb525ba437de428653a1040e2aed7d696640bf75203786a6d7a0e67cfcc",
        "De Cock, D. (2011). Ames, Iowa: Alternative to the Boston housing "
        "data set. Journal of Statistics Education, 19(3). "
        "https://doi.org/10.1080/10691898.2011.11889627",
    ),
    # ------------------------------------------------------------------
    # Belgian NO₂ air quality time series (2000 – 2016)
    # 149 039 hourly measurements at four Brussels monitoring stations
    # ------------------------------------------------------------------
    "20000101_20161231-NO2.csv": (
        "https://raw.githubusercontent.com/paris-saclay-cds/python-workshop"
        "/master/Day_1_Scientific_Python/data/20000101_20161231-NO2.csv",
        "33233638cb3dc7e0004c09b26c7ee2a7b9d9abf0d34dd96579f275b0e67dd19a",
        "Belgian Interregional Environment Agency (IRCELINE) — NO₂ "
        "measurements at BASCH, BONAP, PA18 and VERS stations in Brussels. "
        "Distributed as part of the Paris-Saclay CDS Python Workshop "
        "(https://github.com/paris-saclay-cds/python-workshop).",
    ),
    # ------------------------------------------------------------------
    # French 2005 European Constitution referendum results
    # 36 660 communes × 9 columns
    # ------------------------------------------------------------------
    "referendum.csv": (
        "https://raw.githubusercontent.com/demianw/Xed"
        "/main/data_wrangling/data/referendum.csv",
        "ffb0e560f1044621d6cbc15aaaf701396138cd52ff64a8e2c576e523fbc1f92c",
        "French Ministry of the Interior — Results of the 29 May 2005 "
        "referendum on the Treaty establishing a Constitution for Europe, "
        "by commune.",
    ),
    # ------------------------------------------------------------------
    # French administrative geography: departments and regions
    # ------------------------------------------------------------------
    "departments.csv": (
        "https://raw.githubusercontent.com/demianw/Xed"
        "/main/data_wrangling/data/departments.csv",
        "c4c40dc3f8a67a0d444a76176d46d088dbc55d35b6388d976c959acdc1da1a21",
        "French department codes and names (INSEE 2015). "
        "Served from the Xed course repository.",
    ),
    "regions.csv": (
        "https://raw.githubusercontent.com/demianw/Xed"
        "/main/data_wrangling/data/regions.csv",
        "840783aeb203c0389a89bf8c372ecb31837b8d17fedd2b3f5f3bc13c7cea90b7",
        "French region codes and names (INSEE 2015). "
        "Served from the Xed course repository.",
    ),
    "regions.geojson": (
        "https://raw.githubusercontent.com/demianw/Xed"
        "/main/data_wrangling/data/regions.geojson",
        "84bccb38f31f1e6db7dbe7920373b684e6ea043147e7254c27a290ed95694a14",
        "French metropolitan region boundaries (GeoJSON). "
        "Served from the Xed course repository.",
    ),
    "departements.geojson": (
        "https://raw.githubusercontent.com/demianw/Xed"
        "/main/data_wrangling/data/departements.geojson",
        "a5557a059bd4f848fbf92e81f716d7e2253226f31e943f67f85b844c9de91c5d",
        "French metropolitan department boundaries (GeoJSON). "
        "Served from the Xed course repository.",
    ),
    # ------------------------------------------------------------------
    # US County-level Presidential Election Context (2012 & 2016)
    # ------------------------------------------------------------------
    "election-context-2018.csv": (
        "https://raw.githubusercontent.com/MEDSL/2018-elections-unoffical"
        "/master/election-context-2018.csv",
        "9d04bb5bbbb8821692f8dfee2cfa31f91fc933a458af05e52fd0d3b395286513",
        "Kuriwaki, S., Ansolabehere, S., Dagonel, A., & Yamauchi, S. (2021). "
        "MIT Election Data and Science Lab. "
        "https://github.com/MEDSL/2018-elections-unoffical",
    ),
}


# ---------------------------------------------------------------------------
# Internal download helper
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(name: str, verbose: bool = True) -> Path:
    """
    Return the local path to *name*, downloading it first if necessary.

    Parameters
    ----------
    name : str
        Key in ``_REGISTRY``.
    verbose : bool
        Print progress messages when downloading.

    Returns
    -------
    Path
        Local path to the cached file.

    Raises
    ------
    KeyError
        If *name* is not in ``_REGISTRY``.
    RuntimeError
        If the downloaded file's SHA-256 does not match the registered digest.
    """
    if name not in _REGISTRY:
        raise KeyError(f"{name!r} is not in the dataset registry.")

    url, expected_sha256, citation = _REGISTRY[name]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / name

    # ── Return cached file if it exists and is intact ──────────────────
    if dest.exists():
        actual = _sha256(dest.read_bytes())
        if actual == expected_sha256:
            return dest
        # Cached file is corrupted or was truncated; delete and re-download.
        if verbose:
            print(f"[xed] Cached {name} failed SHA-256 check — re-downloading.")
        dest.unlink()

    # ── Download ────────────────────────────────────────────────────────
    if verbose:
        print(f"[xed] Downloading {name} …")
        print(f"      Source : {url}")
        print(f"      Cache  : {dest}")

    try:
        import pooch  # type: ignore

        pooch.retrieve(
            url=url,
            known_hash=f"sha256:{expected_sha256}",
            fname=name,
            path=_CACHE_DIR,
            progressbar=verbose,
        )
    except ImportError:
        # ── urllib fallback ─────────────────────────────────────────────
        # pooch is not installed; use urllib with a manual integrity check.
        data, _ = urllib.request.urlretrieve(url, dest)
        actual = _sha256(Path(data).read_bytes())
        if actual != expected_sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {name}.\n"
                f"  Expected : {expected_sha256}\n"
                f"  Got      : {actual}\n"
                f"The file may have been modified at the source URL."
            )

    if verbose:
        print(f"[xed] {name} ready.")
    return dest


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_titanic(verbose: bool = True) -> pd.DataFrame:
    """
    Load the Titanic passenger dataset.

    Returns a DataFrame with ``PassengerId`` as the index and the following
    columns: ``Survived``, ``Pclass``, ``Name``, ``Sex``, ``Age``,
    ``SibSp``, ``Parch``, ``Ticket``, ``Fare``, ``Cabin``, ``Embarked``.

    The data is downloaded once and cached at::

        ~/.cache/xed_datasets/titanic.csv

    Parameters
    ----------
    verbose : bool
        Print download progress when the file is not yet cached.

    Returns
    -------
    pd.DataFrame
        891 rows × 11 columns.

    Source
    ------
    British Board of Trade (1912) inquiry into the loss of the SS Titanic.
    Distributed in its current format by datasciencedojo/datasets on GitHub.
    """
    path = _fetch("titanic.csv", verbose=verbose)
    df = pd.read_csv(path, index_col="PassengerId")
    return df


def load_ames_housing(verbose: bool = True) -> pd.DataFrame:
    """
    Load the Ames Iowa Housing dataset.

    Returns a DataFrame with 82 columns including the target ``SalePrice``.
    Column names follow the original DeCode (2011) format with spaces
    (e.g. ``"MS SubClass"``, ``"Lot Frontage"``).

    The data is downloaded once and cached at::

        ~/.cache/xed_datasets/AmesHousing.txt

    Parameters
    ----------
    verbose : bool
        Print download progress when the file is not yet cached.

    Returns
    -------
    pd.DataFrame
        2 930 rows × 82 columns.

    Source
    ------
    De Cock, D. (2011). Ames, Iowa: Alternative to the Boston housing
    data set. Journal of Statistics Education, 19(3).
    https://doi.org/10.1080/10691898.2011.11889627
    """
    path = _fetch("AmesHousing.txt", verbose=verbose)
    df = pd.read_csv(path, sep="\t")
    return df


def load_no2(verbose: bool = True) -> pd.DataFrame:
    """
    Load the Belgian NO₂ air quality time series (2000 – 2016).

    Returns a time-indexed DataFrame with four columns corresponding to
    monitoring stations in Brussels: ``BASCH``, ``BONAP``, ``PA18``, ``VERS``.
    Values are hourly NO₂ concentrations in µg/m³.

    The data is downloaded once and cached at::

        ~/.cache/xed_datasets/20000101_20161231-NO2.csv

    Parameters
    ----------
    verbose : bool
        Print download progress when the file is not yet cached.

    Returns
    -------
    pd.DataFrame
        149 039 rows × 4 columns, DatetimeIndex named ``timestamp``.

    Source
    ------
    Belgian Interregional Environment Agency (IRCELINE).
    Distributed as part of the Paris-Saclay CDS Python Workshop:
    https://github.com/paris-saclay-cds/python-workshop
    """
    path = _fetch("20000101_20161231-NO2.csv", verbose=verbose)
    df = pd.read_csv(
        path,
        sep=";",
        skiprows=[1],           # row 1 contains the unit label (µg/m³)
        na_values=["n/d"],      # station-down / missing value marker
        index_col=0,
        parse_dates=True,
    )
    df.index.name = "timestamp"
    return df


def load_french_referendum(verbose: bool = True) -> pd.DataFrame:
    """
    Load the results of the French 2005 European Constitution referendum.

    Returns a DataFrame with one row per commune and nine columns:
    ``Department code``, ``Department name``, ``Town code``, ``Town name``,
    ``Registered``, ``Abstentions``, ``Null``, ``Choice A`` (Yes), ``Choice B`` (No).

    The data is downloaded once and cached at::

        ~/.cache/xed_datasets/referendum.csv

    Parameters
    ----------
    verbose : bool
        Print download progress when the file is not yet cached.

    Returns
    -------
    pd.DataFrame
        36 660 rows × 9 columns.

    Source
    ------
    French Ministry of the Interior — results of the referendum of
    29 May 2005 on the Treaty establishing a Constitution for Europe.
    """
    path = _fetch("referendum.csv", verbose=verbose)
    return pd.read_csv(path, sep=";")


def load_french_departments(verbose: bool = True) -> pd.DataFrame:
    """
    Load the table of French metropolitan departments.

    Returns a DataFrame with columns: ``id``, ``region_code``, ``code``,
    ``name``, ``slug``.

    Parameters
    ----------
    verbose : bool
        Print download progress when the file is not yet cached.

    Returns
    -------
    pd.DataFrame
        96 rows × 5 columns.

    Source
    ------
    INSEE departmental codes (2015).
    """
    path = _fetch("departments.csv", verbose=verbose)
    return pd.read_csv(path)


def load_french_regions(verbose: bool = True) -> pd.DataFrame:
    """
    Load the table of French metropolitan regions.

    Returns a DataFrame with columns: ``id``, ``code``, ``name``, ``slug``.

    Parameters
    ----------
    verbose : bool
        Print download progress when the file is not yet cached.

    Returns
    -------
    pd.DataFrame
        22 rows × 4 columns.

    Source
    ------
    INSEE region codes (2015).
    """
    path = _fetch("regions.csv", verbose=verbose)
    return pd.read_csv(path)


def fetch_french_geojson(
    level: str = "regions",
    verbose: bool = True,
) -> Path:
    """
    Return the local path to a French administrative boundary GeoJSON file.

    Parameters
    ----------
    level : {'regions', 'departements'}
        Which administrative level to fetch.
    verbose : bool
        Print download progress when the file is not yet cached.

    Returns
    -------
    Path
        Local path to the cached ``.geojson`` file, suitable for
        ``geopandas.read_file()`` or ``json.load()``.

    Source
    ------
    French metropolitan region / department boundaries.
    """
    if level not in ("regions", "departements"):
        raise ValueError(f"level must be 'regions' or 'departements', got {level!r}")
    return _fetch(f"{level}.geojson", verbose=verbose)


# ---------------------------------------------------------------------------
# Convenience summary
# ---------------------------------------------------------------------------

def load_us_county_elections(verbose: bool = True) -> pd.DataFrame:
    """
    Load the MIT MEDSL US county-level election and demographics dataset.

    3 114 US counties × 39 columns: presidential results for 2012 and 2016
    (raw votes), senate/house/governor results where available, and 15 ACS 2015
    demographic features (``white_pct``, ``black_pct``, ``hispanic_pct``,
    ``median_hh_inc``, ``rural_pct``, etc.).

    Cached at ``~/.cache/xed_datasets/election-context-2018.csv``.

    Source: MIT Election Data and Science Lab (MEDSL), Kuriwaki et al. (2021).
    https://github.com/MEDSL/2018-elections-unoffical
    """
    path = _fetch("election-context-2018.csv", verbose=verbose)
    return pd.read_csv(path)


def show_registry() -> None:
    """Print all registered datasets with their sources."""
    print(f"{'Name':<35} {'Size':>6}  URL")
    print("-" * 90)
    for name, (url, sha256, citation) in _REGISTRY.items():
        dest = _CACHE_DIR / name
        size = f"{dest.stat().st_size // 1024}KB" if dest.exists() else "—"
        print(f"{name:<35} {size:>6}  {url[:50]}")
