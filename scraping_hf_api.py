
import requests
import time
import datetime as dt
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE = "https://huggingface.co"
ENDPOINT = f"{BASE}/api/daily_papers"

HEADERS = {
    # 実利用の UA を設定（Bot 誤検知を避け、CDN キャッシュも恩恵を受けやすい）
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

def fetch_daily_papers(date_str: str, limit: int = 20, max_pages: int = 100) -> List[Dict[str, Any]]:
    """
    指定日付(YYYY-MM-DD)の daily_papers を全ページ回収する。
    429 対策で指数バックオフ付き。
    """
    all_items: List[Dict[str, Any]] = []
    page = 1
    session = requests.Session()

    while page <= max_pages:
        params = {"date": date_str, "page": page, "limit": limit}
        backoff = 5  # seconds
        max_retries = 6

        for attempt in range(max_retries):
            resp = session.get(ENDPOINT, headers=HEADERS, params=params, timeout=30,verify=False)  # verify=True(既定)
            if resp.status_code == 429:
                # レートリミット: バックオフ
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()  # その他 4xx/5xx は例外
            data = resp.json()
            break
        else:
            raise RuntimeError(f"Max retries exceeded for {params}")

        items = data if isinstance(data, list) else data.get("items") or data.get("data") or []
        if not items:
            # これ以上ページ無し
            break

        all_items.extend(items)
        page += 1
        # 過度な連続アクセスを避ける（HF は短時間過多で 429 が発生しやすい）[3](https://huggingface.co/papers/trending)[4](https://api.endpoints.huggingface.cloud/)
        time.sleep(1.0)

    return all_items

def to_csv_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    JSON から CSV に落としやすいフィールドを抜粋・整形。
    スキーマは実データを観察して適宜拡張してください（title, arxiv_id, url, authors, tags, upvotes 等）。
    """
    rows = []
    for it in items:
        rows.append({
            "title": it.get("title"),
            "arxiv_id": it.get("arxiv_id") or it.get("arxivId"),
            "arxiv_url": it.get("arxiv_url") or it.get("arxivUrl"),
            "hf_paper_url": it.get("url") or it.get("paper_url"),
            "submitted_by": (it.get("submitted_by") or it.get("submittedBy") or {}).get("name")
                            if isinstance(it.get("submitted_by") or it.get("submittedBy"), dict)
                            else it.get("submitted_by") or it.get("submittedBy"),
            "upvotes": it.get("upvotes") or it.get("score"),
            "published_at": it.get("published_at") or it.get("publishedAt"),
            "created_at": it.get("created_at") or it.get("createdAt"),
        })
    return rows

def save_json(items: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def save_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # 空でもヘッダーを出しておくと集計が楽
        fieldnames = ["title","arxiv_id","arxiv_url","hf_paper_url","submitted_by","upvotes","published_at","created_at"]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    # 例：前日分を収集（日本時間ベース）
    target_date = (dt.datetime.now().date() - dt.timedelta(days=1)).strftime("%Y-%m-%d")

    items = fetch_daily_papers(target_date, limit=20, max_pages=100)
    rows = to_csv_rows(items)

    out_dir = Path("data") / target_date
    save_json(items, out_dir / f"daily_papers_{target_date}.json")
    save_csv(rows, out_dir / f"daily_papers_{target_date}.csv")

    print(f"Collected {len(items)} items for {target_date}")
