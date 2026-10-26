"""TDC ADMET web leaderboard scrape + 24h cache.

Moved fetch_leaderboard from the old ``compare_leaderboard.py:62-153`` into the package.
Works with cache alone if there's no internet in the user's environment (or returns an empty dict).

Why:
    Avoid pulling requests/bs4/numpy/pandas into sys.modules at import-time;
    all dependencies are lazy-imported inside functions. Just importing the evaluation package
    must not auto-register requests in sys.modules (same as other lazy invariants).

Note:
    The leaderboard depends on TDC's static webpage structure. If TDC changes the page, this
    parser can break — in that case ``fetch_leaderboard`` should just return an empty dict,
    so callers handle it gracefully without affecting the experiment results themselves.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

# DATASET → TDC page slug. Same mapping as the old compare_leaderboard.py.
DATASET_URLS: dict[str, str] = {
    "Caco2_Wang": "01caco2",
    "HIA_Hou": "02hia",
    "Pgp_Broccatelli": "03pgp",
    "Bioavailability_Ma": "04bioav",
    "Lipophilicity_AstraZeneca": "05lipo",
    "Solubility_AqSolDB": "06aqsol",
    "BBB_Martins": "07bbb",
    "PPBR_AZ": "08ppbr",
    "VDss_Lombardo": "09vdss",
    "CYP2C9_Veith": "10cyp2c9i",
    "CYP2D6_Veith": "11cyp2d6i",
    "CYP3A4_Veith": "12cyp3a4i",
    "CYP2C9_Substrate_CarbonMangels": "13cyp2c9s",
    "CYP2D6_Substrate_CarbonMangels": "14cyp2d6s",
    "CYP3A4_Substrate_CarbonMangels": "15cyp3a4s",
    "Half_Life_Obach": "16halflife",
    "Clearance_Hepatocyte_AZ": "17clhepa",
    "Clearance_Microsome_AZ": "18clmicro",
    "LD50_Zhu": "19ld50",
    "hERG": "20herg",
    "AMES": "21ames",
    "DILI": "22dili",
}

LOWER_IS_BETTER = {"mae"}
BASE_URL = "https://tdcommons.ai/benchmark/admet_group"
DEFAULT_CACHE_TTL_SEC = 86400  # 24h


def default_cache_path() -> Path:
    """``QADMET_LEADERBOARD_CACHE`` env takes priority; otherwise a hidden file next to artifacts."""
    env = os.environ.get("QADMET_LEADERBOARD_CACHE")
    if env:
        return Path(env).resolve()
    from qadmet.core.paths import artifacts_dir

    return artifacts_dir() / ".leaderboard_cache.json"


def _parse_leaderboard_page(url: str) -> tuple[dict[str, dict[str, float]], str, str]:
    """Parses a single dataset page.

    Returns:
        (models_dict, metric_name, direction). models_dict is
        ``{model_name: {"score": float, "std": float}}``.
        Returns (``{}, "", ""``) on parse failure.
    """
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    tables = soup.find_all("table")
    if len(tables) < 2:
        return {}, "", ""

    lb_table = tables[1]
    headers = [
        th.get_text(strip=True)
        for th in lb_table.find_all("tr")[0].find_all(["th", "td"])
    ]
    metric_col = headers[-1] if headers else ""
    metric_name = metric_col.lower()

    results: dict[str, dict[str, float]] = {}
    for row in lb_table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 2:
            continue
        model_name = cells[1]
        score_str = cells[-1]
        match = re.match(r"([0-9.]+)\s*[±+-]+\s*([0-9.]+)", score_str)
        if match:
            score = float(match.group(1))
            std = float(match.group(2))
        else:
            try:
                score = float(score_str)
                std = 0.0
            except ValueError:
                continue
        results[model_name] = {"score": score, "std": std}

    direction = "lower" if metric_name in LOWER_IS_BETTER else "higher"
    return results, metric_name, direction


def fetch_leaderboard(
    datasets: list[str] | None = None,
    *,
    use_cache: bool = True,
    cache_path: Path | None = None,
    cache_ttl_sec: int = DEFAULT_CACHE_TTL_SEC,
    sleep_sec: float = 0.3,
) -> dict[str, dict[str, Any]]:
    """Parses the ADMET leaderboard from the TDC website.

    Args:
        datasets: Dataset names to fetch (DATASET_URLS keys). All 22 if None.
        use_cache: If True, no network call is made while within the cache TTL.
        cache_path: Cache JSON path (default_cache_path if None).
        cache_ttl_sec: Cache validity period.
        sleep_sec: Sleep between pages (rate-limit protection).

    Returns:
        ``{dataset_name: {"metric": str, "direction": str, "models": {model: {...}}}}``.
        Datasets that encounter network failures are skipped.
    """
    cache_path = cache_path or default_cache_path()
    if use_cache and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            age = time.time() - cache.get("_timestamp", 0)
            if age < cache_ttl_sec:
                data = {k: v for k, v in cache.items() if k != "_timestamp"}
                if datasets is not None:
                    data = {k: v for k, v in data.items() if k in datasets}
                return data
        except (json.JSONDecodeError, OSError):  # pragma: no cover
            pass

    if datasets is None:
        datasets = list(DATASET_URLS.keys())

    leaderboard: dict[str, dict[str, Any]] = {}
    for dataset_name in datasets:
        slug = DATASET_URLS.get(dataset_name)
        if slug is None:
            continue
        url = f"{BASE_URL}/{slug}/"
        try:
            models, metric, direction = _parse_leaderboard_page(url)
            leaderboard[dataset_name] = {
                "metric": metric,
                "direction": direction,
                "models": models,
            }
        except Exception:  # noqa: BLE001  — network/parsing failures are skipped per-dataset.
            continue
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({**leaderboard, "_timestamp": time.time()}, indent=2))
    return leaderboard


def get_rank(models: dict[str, dict[str, float]], score: float, direction: str) -> tuple[int, int]:
    """Expected rank (1-indexed) of the score on the leaderboard + total number of models.

    Args:
        models: ``{model_name: {"score": float, "std": float}}``.
        score: Our model's score.
        direction: ``"lower"`` (lower is better, e.g. mae) | ``"higher"``.

    Returns:
        (rank, total). Ties are defined as the number of models strictly better than this one + 1.
    """
    scores = [m["score"] for m in models.values()]
    n_total = len(scores)
    if direction == "lower":
        rank = sum(1 for s in scores if s < score) + 1
    else:
        rank = sum(1 for s in scores if s > score) + 1
    return rank, n_total


def compare_to_leaderboard(
    metric_df,  # pandas.DataFrame; type annotation only as a string for laziness.
    *,
    methods: list[str] | None = None,
    use_cache: bool = True,
    cache_path: Path | None = None,
):
    """For each task in the target metric DataFrame, the leaderboard rank and comparison model scores
    Returns the combined DataFrame.

    Args:
        metric_df: Output of ``per_task_metric_df`` (or the same column shape).
        methods: Leaderboard model names to compare (e.g. ``["Chemprop", "MapLight + GNN"]``).
            The special key ``"Top"`` refers to the 1st place model + score for each task.
        use_cache / cache_path: Same as ``fetch_leaderboard``.

    Returns:
        ``[dataset, metric, ours, ours_std, rank, n_total, *method_score_cols]``
        DataFrame. Tasks without leaderboard data have NaN rank/methods.
    """
    import pandas as pd

    methods = list(methods or [])
    datasets = list(metric_df["dataset"]) if "dataset" in metric_df.columns else []
    leaderboard = fetch_leaderboard(
        datasets=datasets, use_cache=use_cache, cache_path=cache_path
    )

    rows: list[dict[str, Any]] = []
    for _, r in metric_df.iterrows():
        ds = str(r["dataset"])
        ours = float(r["mean"]) if "mean" in r.index else float("nan")
        ours_std = float(r["std"]) if "std" in r.index else float("nan")
        lb = leaderboard.get(ds)
        row: dict[str, Any] = {
            "dataset": ds,
            "metric": str(r["metric"]),
            "ours": ours,
            "ours_std": ours_std,
            "rank": float("nan"),
            "n_total": 0,
        }
        if lb is not None and lb.get("models"):
            rank, n_total = get_rank(lb["models"], ours, lb["direction"])
            row["rank"] = int(rank)
            row["n_total"] = int(n_total)
            for m in methods:
                if m == "Top":
                    if lb["direction"] == "lower":
                        top_model = min(lb["models"], key=lambda k: lb["models"][k]["score"])
                    else:
                        top_model = max(lb["models"], key=lambda k: lb["models"][k]["score"])
                    row["Top"] = lb["models"][top_model]["score"]
                    row["Top_model"] = top_model
                else:
                    md = lb["models"].get(m)
                    row[m] = float(md["score"]) if md is not None else float("nan")
        else:
            for m in methods:
                row[m if m != "Top" else "Top"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
