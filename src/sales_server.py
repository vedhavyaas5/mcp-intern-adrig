from __future__ import annotations

import csv
import json
import os
import random
import re
import string
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne

# NOTE: MCP CLI tools (e.g. `mcp dev file.py`) auto-detect a global named
# `mcp`, `server`, or `app`. Keep this as `mcp` for compatibility.
mcp = FastMCP("Sales-System")

# Load environment variables from a local .env when present.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


def _get_mongo_collection():
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")

    db_name = os.environ.get("MONGODB_DB", "intern_adrig")
    collection_name = os.environ.get("MONGODB_COLLECTION", "shopping_trends")

    client = MongoClient(uri)
    return client[db_name][collection_name]


def _to_bool(value: str) -> bool | None:
    v = (value or "").strip().lower()
    if v == "yes":
        return True
    if v == "no":
        return False
    return None


def _normalize_shopping_trend_row(row: dict[str, str]) -> dict[str, Any]:
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


MOCK_PRODUCT_CATALOG = [
    {
        "sku": "PROD-AIDA-01",
        "title": "Nexus Traffic-Vision Monitoring Suite",
        "description": "Computer-vision vehicle classification engine optimized for urban traffic hubs. Analyzes raw video feeds, counts vehicles, and predicts congestions using deep learning models.",
        "price": "₹25,000/month",
    },
    {
        "sku": "PROD-FIN-02",
        "title": "Smart-Spend Predictive Ledger",
        "description": "Time-series prediction engine for corporate expense analytics. Automatically groups expenses, maps seasonal trends, and forecasts quarterly institutional savings using advanced regression models.",
        "price": "₹12,000/month",
    },
]


@mcp.tool()
def query_product_catalog(search_intent: str) -> str:
    """Keyword lookup over the mock 'Elasticsearch-like' product catalog."""

    keywords = search_intent.lower().split()
    results: list[str] = []

    for product in MOCK_PRODUCT_CATALOG:
        searchable = (product["title"] + " " + product["description"]).lower()
        if any(word in searchable for word in keywords):
            results.append(
                f"📦 Product: {product['title']} [{product['sku']}]\n"
                f"Price: {product['price']}\n"
                f"Summary: {product['description']}"
            )

    if results:
        return "\n\n".join(results)

    return "⚠️ Catalog Notice: 0 matching products found."


@mcp.tool()
def generate_checkout_incentive(context: str = "") -> str:
    """Generate a simple discount code to help close a lead."""

    suffix = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    code = f"AIDA10-{suffix}"
    base = (
        "✅ Checkout incentive generated.\n"
        f"Code: {code}\n"
        "Discount: 10% off first month"
    )
    if context:
        return f"{base}\nContext: {context}"
    return base


@mcp.tool()
def import_shopping_trends_csv_to_mongodb(
    csv_path: str = "dataset/shopping_trends.csv",
) -> str:
    """Import dataset/shopping_trends.csv into MongoDB (upsert by customer_id)."""

    path = Path(csv_path)
    if not path.exists():
        return f"❌ Import failed: CSV file not found: {path}"

    try:
        collection = _get_mongo_collection()
    except Exception as exc:
        return f"❌ MongoDB connection error: {exc}"

    ops: list[UpdateOne] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                doc = _normalize_shopping_trend_row(row)
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
    except Exception as exc:
        return f"❌ Import failed: {exc}"

    return f"✅ Imported/updated {len(ops)} rows into MongoDB collection '{collection.name}'."


@mcp.tool()
def get_shopping_trend_by_customer_id(customer_id: int) -> str:
    """Fetch a single shopping trends record by customer_id from MongoDB."""

    try:
        collection = _get_mongo_collection()
        doc = collection.find_one({"customer_id": int(customer_id)}, {"_id": 0})
    except Exception as exc:
        return f"❌ MongoDB query error: {exc}"

    if not doc:
        return f"⚠️ No record found for customer_id={customer_id}." 

    lines = [f"✅ Shopping Trend Record (customer_id={customer_id})"]
    for key, value in doc.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


@mcp.tool()
def summarize_shopping_trends(
    field: str = "",
    limit: int = 10,
    group_by: str | None = None,
    aggregate: str | None = None,
    aggregate_field: str | None = None,
    aggregations: list[dict[str, Any]] | None = None,
    sort_by: str | None = None,
    sort: str = "desc",
    filter: dict[str, Any] | None = None,
) -> str:
    """Group-by summary with optional multi-aggregations.

    Supports:
    - group_by + aggregations (list of {op, field, alias})
    - legacy aggregate/aggregate_field
    - sort_by, sort, limit, filter
    """

    group_key = (group_by or field or "").strip()
    agg_list = aggregations or []
    flt = filter or {}

    if group_key and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", group_key):
        return "❌ Invalid group_by field name. Use letters/numbers/underscore only."

    # Legacy single-aggregate fallback
    if not agg_list and aggregate:
        op = str(aggregate).strip().lower()
        alias = f"{op}_{aggregate_field or group_key or 'count'}"
        agg_list = [{"op": op, "field": aggregate_field, "alias": alias}]

    # Default to count if nothing specified
    if not agg_list:
        if not group_key:
            return "❌ Missing group_by field for summary."
        agg_list = [{"op": "count", "alias": "total_purchases"}]

    for agg in agg_list:
        op = str(agg.get("op") or "").strip().lower()
        if op not in {"count", "avg", "sum", "min", "max"}:
            return "❌ Invalid aggregation op. Use avg, sum, count, min, max."
        field_name = str(agg.get("field") or "").strip()
        if op != "count" and field_name and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field_name):
            return "❌ Invalid aggregation field name."

    # Prevent accidental high-cardinality / huge output.
    try:
        capped_limit = max(1, min(int(limit), 50))
    except Exception:
        capped_limit = 10

    try:
        collection = _get_mongo_collection()

        group_stage: dict[str, Any] = {"_id": f"${group_key}"}
        for agg in agg_list:
            op = str(agg.get("op") or "").strip().lower()
            alias = str(agg.get("alias") or f"{op}_{agg.get('field') or 'count'}").strip()
            field_name = str(agg.get("field") or "").strip()

            if op == "count":
                group_stage[alias] = {"$sum": 1}
            elif op == "avg" and field_name:
                group_stage[alias] = {"$avg": f"${field_name}"}
            elif op == "sum" and field_name:
                group_stage[alias] = {"$sum": f"${field_name}"}
            elif op == "min" and field_name:
                group_stage[alias] = {"$min": f"${field_name}"}
            elif op == "max" and field_name:
                group_stage[alias] = {"$max": f"${field_name}"}

        sort_alias = str(sort_by or "").strip()
        if not sort_alias and agg_list:
            sort_alias = str(agg_list[0].get("alias") or "").strip()
        sort_dir = DESCENDING if str(sort or "desc").lower() == "desc" else ASCENDING
        sort_stage = {sort_alias: sort_dir} if sort_alias else {"_id": sort_dir}

        pipeline: list[dict[str, Any]] = []
        if flt:
            pipeline.append({"$match": flt})
        pipeline.append({"$group": group_stage})
        pipeline.append({"$sort": sort_stage})
        pipeline.append({"$limit": capped_limit})

        rows = list(collection.aggregate(pipeline))
    except Exception as exc:
        return f"❌ MongoDB aggregate error: {exc}"

    if not rows:
        return "⚠️ No data found (did you import the CSV yet?)."

    results: list[dict[str, Any]] = []
    for doc in rows:
        row = {group_key: doc.get("_id")}
        for key, value in doc.items():
            if key == "_id":
                continue
            row[key] = round(value, 2) if isinstance(value, float) else value
        results.append(row)

    return json.dumps(results, ensure_ascii=True)


@mcp.tool()
def query_shopping_trends(
    query_json: str = "{}",
    limit: int = 10,
    count_only: bool = False,
    sort_by: str | None = None,
    sort: str = "desc",
    projection: list[str] | None = None,
) -> str:
    """Query the shopping trends MongoDB collection using a JSON query selector.

    Fields available: customer_id (int), age (int), gender (str), item_purchased (str),
    category (str), purchase_amount_usd (float), location (str), size (str), color (str),
    season (str), review_rating (float), subscription_status (bool), payment_method (str),
    shipping_type (str), discount_applied (bool), promo_code_used (bool),
    previous_purchases (int), preferred_payment_method (str), frequency_of_purchases (str).

    Example query_json: '{"color": "White"}' or '{"category": "Footwear", "season": "Winter"}'

    If count_only=True, returns the total count of matching records.
    Optional: sort_by, sort (asc|desc), projection (list of fields).
    """
    import json
    try:
        filter_dict = json.loads(query_json)
    except Exception as exc:
        return f"❌ Invalid JSON query: {exc}"

    # Normalize keys/values in filter_dict to match dataset casing if needed,
    # or rely on the LLM to format it correctly based on description/schema.
    # To be safe, if we find color/item_purchased values, we make sure they are capitalized.
    # E.g. {"color": "white"} -> {"color": "White"}
    normalized_filter = {}
    for k, v in filter_dict.items():
        k_str = str(k).strip().lower()
        # Map lowercase key to exact Mongo field name
        key_mapping = {
            "customer_id": "customer_id",
            "customer id": "customer_id",
            "customerid": "customer_id",
            "age": "age",
            "gender": "gender",
            "item_purchased": "item_purchased",
            "item purchased": "item_purchased",
            "category": "category",
            "purchase_amount_usd": "purchase_amount_usd",
            "purchase amount": "purchase_amount_usd",
            "purchase_amount": "purchase_amount_usd",
            "location": "location",
            "size": "size",
            "color": "color",
            "season": "season",
            "review_rating": "review_rating",
            "review rating": "review_rating",
            "subscription_status": "subscription_status",
            "subscription status": "subscription_status",
            "payment_method": "payment_method",
            "payment method": "payment_method",
            "shipping_type": "shipping_type",
            "shipping type": "shipping_type",
            "discount_applied": "discount_applied",
            "discount applied": "discount_applied",
            "promo_code_used": "promo_code_used",
            "promo code used": "promo_code_used",
            "previous_purchases": "previous_purchases",
            "previous purchases": "previous_purchases",
            "preferred_payment_method": "preferred_payment_method",
            "preferred payment method": "preferred_payment_method",
            "frequency_of_purchases": "frequency_of_purchases"
        }
        actual_key = key_mapping.get(k_str, k)
        
        # Capitalize values if they are strings and match fields like color, location, category, item_purchased, season, gender
        if isinstance(v, str) and actual_key in {"color", "location", "category", "item_purchased", "season", "gender"}:
            # Match casing of dataset values (e.g. "white" -> "White")
            v = v.strip().title()
        
        normalized_filter[actual_key] = v

    def _normalize_field_key(raw_key: str) -> str:
        cleaned = (raw_key or "").strip().lower()
        return key_mapping.get(cleaned, raw_key)

    try:
        capped_limit = max(1, min(int(limit), 50))
    except Exception:
        capped_limit = 10

    try:
        collection = _get_mongo_collection()
        if count_only:
            total = collection.count_documents(normalized_filter)
            if total == 0:
                return "⚠️ No matching records found in shopping trends database."
            return f"✅ Counted {total} matching records in shopping trends."

        projection_doc: dict[str, int] = {"_id": 0}
        if projection:
            projection_doc = {str(field): 1 for field in projection if str(field).strip()}
            projection_doc["_id"] = 0

        cursor = collection.find(normalized_filter, projection_doc)

        sort_key = _normalize_field_key(sort_by or "").strip()
        if sort_key:
            direction = DESCENDING if str(sort or "desc").lower() == "desc" else ASCENDING
            cursor = cursor.sort(sort_key, direction)

        docs = list(cursor.limit(capped_limit))
    except Exception as exc:
        return f"❌ MongoDB query error: {exc}"

    if not docs:
        return "⚠️ No matching records found in shopping trends database."

    out = [f"✅ Found {len(docs)} matching records in shopping trends:"]
    for i, doc in enumerate(docs, 1):
        out.append(f"\nRecord #{i}:")
        for key, val in doc.items():
            out.append(f"- {key}: {val}")
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run(transport="stdio")
