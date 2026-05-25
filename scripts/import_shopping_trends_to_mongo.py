from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "dataset" / "shopping_trends.csv"


def _to_bool(value: str) -> bool | None:
    v = (value or "").strip().lower()
    if v == "yes":
        return True
    if v == "no":
        return False
    return None


def _normalize_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "customer_id": int(row["Customer ID"]),
        "age": int(row["Age"]),
        "gender": row["Gender"],
        "item_purchased": row["Item Purchased"],
        "category": row["Category"],
        "purchase_amount_usd": float(row["Purchase Amount (USD)"]),
        "location": row["Location"],
        "size": row["Size"],
        "color": row["Color"],
        "season": row["Season"],
        "review_rating": float(row["Review Rating"]),
        "subscription_status": _to_bool(row["Subscription Status"]),
        "payment_method": row["Payment Method"],
        "shipping_type": row["Shipping Type"],
        "discount_applied": _to_bool(row["Discount Applied"]),
        "promo_code_used": _to_bool(row["Promo Code Used"]),
        "previous_purchases": int(row["Previous Purchases"]),
        "preferred_payment_method": row["Preferred Payment Method"],
        "frequency_of_purchases": row["Frequency of Purchases"],
    }


def import_csv_to_mongo(csv_path: Path = DEFAULT_CSV_PATH) -> int:
    # Allow configuration via a local .env file (same pattern as cockpit_api.py)
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")

    db_name = os.environ.get("MONGODB_DB", "intern_adrig")
    collection_name = os.environ.get("MONGODB_COLLECTION", "shopping_trends")

    client = MongoClient(uri)
    collection = client[db_name][collection_name]

    ops: list[UpdateOne] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc = _normalize_row(row)
            ops.append(
                UpdateOne(
                    {"customer_id": doc["customer_id"]},
                    {"$set": doc},
                    upsert=True,
                )
            )

    if ops:
        collection.bulk_write(ops, ordered=False)

    collection.create_index("customer_id", unique=True)

    return len(ops)


def main() -> None:
    csv_path_str = os.environ.get("SHOPPING_TRENDS_CSV")
    csv_path = Path(csv_path_str) if csv_path_str else DEFAULT_CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    count = import_csv_to_mongo(csv_path)
    print(f"Imported/updated {count} rows into MongoDB.")


if __name__ == "__main__":
    main()
