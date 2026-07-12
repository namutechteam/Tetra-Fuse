"""TDC ADMET 웹 리더보드 스크레이프 + 24h 캐시.

옛 ``compare_leaderboard.py:62-153`` 의 fetch_leaderboard 를 패키지 안으로 이동.
사용자 환경에서 인터넷이 없으면 cache 만으로도 동작 (또는 빈 dict 반환).

Why:
    sys.modules 에 requests/bs4/numpy/pandas 를 import-time 으로 끌고 들어가지
    않도록 모든 의존을 함수 내부 lazy import. evaluation 패키지를 가져온다고
    requests 가 sys.modules 에 자동 등록되면 안 됨 (다른 lazy invariant 와 동일).

Note:
    리더보드는 TDC 의 정적 웹페이지 구조에 의존. TDC 가 페이지를 바꾸면 본
    파서가 깨질 수 있다 — 그 때는 ``fetch_leaderboard`` 가 빈 dict 를 반환할
    뿐 실험 결과 자체에는 영향이 없도록 호출자는 graceful 처리한다.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

# DATASET → TDC 페이지 슬러그. 옛 compare_leaderboard.py 의 매핑 그대로.
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
    """``QADMET_LEADERBOARD_CACHE`` env 우선, 없으면 artifacts 옆 숨김 파일."""
    env = os.environ.get("QADMET_LEADERBOARD_CACHE")
    if env:
        return Path(env).resolve()
    from qadmet.core.paths import artifacts_dir

    return artifacts_dir() / ".leaderboard_cache.json"


def _parse_leaderboard_page(url: str) -> tuple[dict[str, dict[str, float]], str, str]:
    """한 데이터셋 페이지를 파싱.

    Returns:
        (models_dict, metric_name, direction). models_dict 는
        ``{model_name: {"score": float, "std": float}}``.
        파싱 실패 시 (``{}, "", "")`` 반환.
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
    """TDC 웹사이트에서 ADMET 리더보드를 파싱.

    Args:
        datasets: 가져올 데이터셋 이름 (DATASET_URLS 키). None 이면 전체 22.
        use_cache: True 면 캐시 TTL 안이면 네트워크 호출 안 함.
        cache_path: 캐시 JSON 경로 (None 이면 default_cache_path).
        cache_ttl_sec: 캐시 유효 기간.
        sleep_sec: 페이지 사이 sleep (rate-limit 보호).

    Returns:
        ``{dataset_name: {"metric": str, "direction": str, "models": {model: {...}}}}``.
        네트워크 실패가 발생한 데이터셋은 스킵.
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
        except Exception:  # noqa: BLE001  — 네트워크/파싱 실패는 데이터셋 단위로 스킵.
            continue
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({**leaderboard, "_timestamp": time.time()}, indent=2))
    return leaderboard


def get_rank(models: dict[str, dict[str, float]], score: float, direction: str) -> tuple[int, int]:
    """리더보드에서 점수의 예상 순위 (1-indexed) + 전체 모델 수.

    Args:
        models: ``{model_name: {"score": float, "std": float}}``.
        score: 우리 모델 점수.
        direction: ``"lower"`` (mae 등 작을수록 좋음) | ``"higher"``.

    Returns:
        (rank, total). 동률은 본인보다 명백히 좋은 모델 수 + 1 로 정의.
    """
    scores = [m["score"] for m in models.values()]
    n_total = len(scores)
    if direction == "lower":
        rank = sum(1 for s in scores if s < score) + 1
    else:
        rank = sum(1 for s in scores if s > score) + 1
    return rank, n_total


def compare_to_leaderboard(
    metric_df,  # pandas.DataFrame; lazy 위해 type annotation 만 문자열.
    *,
    methods: list[str] | None = None,
    use_cache: bool = True,
    cache_path: Path | None = None,
):
    """target metric DataFrame 의 각 task 에 대해 리더보드 순위·비교 모델 점수를
    합친 DataFrame 을 반환.

    Args:
        metric_df: ``per_task_metric_df`` 출력 (또는 동일 컬럼 모양).
        methods: 비교할 leaderboard model 이름 (예: ``["Chemprop", "MapLight + GNN"]``).
            ``"Top"`` 이라는 특수 키는 각 task 의 1 위 모델 + 점수.
        use_cache / cache_path: ``fetch_leaderboard`` 와 동일.

    Returns:
        ``[dataset, metric, ours, ours_std, rank, n_total, *method_score_cols]``
        DataFrame. 리더보드 데이터가 없는 task 는 rank/methods 가 NaN.
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
