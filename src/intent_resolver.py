from __future__ import annotations

import re
from typing import Any, Dict, Optional

_FIELD_SYNONYMS: Dict[str, str] = {
    "rating": "review_rating",
    "rated": "review_rating",
    "review": "review_rating",
    "reviews": "review_rating",
    "star": "review_rating",
    "stars": "review_rating",
    "score": "review_rating",
    "price": "purchase_amount_usd",
    "amount": "purchase_amount_usd",
    "cost": "purchase_amount_usd",
    "revenue": "purchase_amount_usd",
    "spend": "purchase_amount_usd",
    "colour": "color",
    "colored": "color",
    "coloured": "color",
    "payment": "payment_method",
    "shipping": "shipping_type",
    "subscription": "subscription_status",
    "promo": "promo_code_used",
    "discount": "discount_applied",
    "customer": "customer_id",
    "age": "age",
    "gender": "gender",
    "category": "category",
    "categories": "category",
    "location": "location",
    "size": "size",
    "color": "color",
    "season": "season",
    "item": "item_purchased",
    "items": "item_purchased",
    "product": "item_purchased",
    "products": "item_purchased",
}

_TOP_HINTS = ("top", "best", "highest", "most", "latest", "newest", "popular", "best selling")
_BOTTOM_HINTS = ("bottom", "worst", "lowest", "least", "minimum", "min", "cheapest", "cheap")
_RATED_HINTS = ("rated", "rating", "review", "stars", "star", "score")
_COUNT_HINTS = ("how many", "count", "number of", "total")
_GROUP_HINTS = ("group by", "by ", "per ", "most common", "top categories", "top category")
_AVG_HINTS = ("average", "avg", "mean")
_MIN_HINTS = ("minimum", "min", "lowest")
_MAX_HINTS = ("maximum", "max", "highest")
_PRODUCT_HINTS = ("item", "items", "product", "products")

_CATEGORY_WORDS = {
    "accessories": "Accessories",
    "footwear": "Footwear",
    "outerwear": "Outerwear",
    "clothing": "Clothing",
    "shoes": "Footwear",
    "shoe": "Footwear",
    "jacket": "Outerwear",
    "jackets": "Outerwear",
}

_COLORS = (
    "white", "black", "red", "blue", "green", "yellow", "orange",
    "purple", "pink", "brown", "gray", "grey", "beige", "cream", "gold", "silver",
)
_SEASONS = ("winter", "spring", "summer", "fall", "autumn")


def _parse_limit(text: str, default: int = 10) -> int:
    m = re.search(r"\btop\s+(\d{1,2})\b", text, flags=re.IGNORECASE)
    if m:
        return max(1, min(int(m.group(1)), 50))
    m = re.search(r"\b(\d{1,2})\s+(?:best|top)\b", text, flags=re.IGNORECASE)
    if m:
        return max(1, min(int(m.group(1)), 50))
    m = re.search(r"\blimit\s+(\d{1,2})\b", text, flags=re.IGNORECASE)
    if m:
        return max(1, min(int(m.group(1)), 50))
    return default


def _find_numeric_field_near_keyword(text: str, keyword: str) -> str:
    around = text.lower()
    if re.search(
        rf"(rating|rated|review|stars?|score)\s+(is\s+)?{re.escape(keyword)}\s+\d+(?:\.\d+)?",
        around,
    ):
        return "review_rating"
    if re.search(
        rf"{re.escape(keyword)}\s+\d+(?:\.\d+)?\s*(rating|rated|review|stars?|score)",
        around,
    ):
        return "review_rating"
    if re.search(
        rf"(price|cost|amount|usd)\s+(is\s+)?{re.escape(keyword)}\s+\d+(?:\.\d+)?",
        around,
    ):
        return "purchase_amount_usd"
    return "purchase_amount_usd"


def _detect_metric_field(text: str, schema: Dict[str, Any]) -> Optional[str]:
    lowered = text.lower()
    for token, field in _FIELD_SYNONYMS.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            if not schema or field in schema:
                return field
    for field in schema:
        if field.replace("_", " ") in lowered or field in lowered:
            return field
    return None


def _extract_group_by(text: str, schema: Dict[str, Any]) -> Optional[str]:
    lowered = text.lower()
    explicit_map = {
        "category": "category",
        "categories": "category",
        "color": "color",
        "colour": "color",
        "season": "season",
        "location": "location",
        "gender": "gender",
        "payment": "payment_method",
        "payment method": "payment_method",
        "shipping": "shipping_type",
        "item": "item_purchased",
        "items": "item_purchased",
        "product": "item_purchased",
        "products": "item_purchased",
    }
    for token, field in explicit_map.items():
        if f"by {token}" in lowered or f"per {token}" in lowered:
            if not schema or field in schema:
                return field
    if "top categories" in lowered or "top category" in lowered:
        return "category"
    if "most common color" in lowered or "most common colour" in lowered:
        return "color"
    return None


def _extract_filters(text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    lowered = text.lower()
    filters: Dict[str, Any] = {}

    for color in _COLORS:
        if re.search(rf"\b{re.escape(color)}\b", lowered):
            filters["color"] = "Gray" if color == "grey" else color.title()
            break

    for season in _SEASONS:
        if re.search(rf"\b{re.escape(season)}\b", lowered):
            filters["season"] = "Fall" if season == "autumn" else season.title()
            break

    for word, cat in _CATEGORY_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            filters["category"] = cat
            break

    between_match = re.search(
        r"\bbetween\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\b", lowered
    )
    if between_match:
        field = _find_numeric_field_near_keyword(lowered, "between")
        lo = float(between_match.group(1))
        hi = float(between_match.group(2))
        filters[field] = {"$gte": min(lo, hi), "$lte": max(lo, hi)}

    under_match = re.search(r"\b(?:under|below|less than)\s+(\d+(?:\.\d+)?)\b", lowered)
    if under_match:
        field = _find_numeric_field_near_keyword(lowered, "under")
        filters[field] = {"$lte": float(under_match.group(1))}

    above_match = re.search(r"\b(?:above|over|greater than|more than)\s+(\d+(?:\.\d+)?)\b", lowered)
    if above_match:
        field = _find_numeric_field_near_keyword(lowered, "above")
        existing = filters.get(field) if isinstance(filters.get(field), dict) else {}
        existing["$gte"] = float(above_match.group(1))
        filters[field] = existing

    if (
        ("top rated" in lowered or "best rated" in lowered or "highest rated" in lowered)
        and "review_rating" not in filters
    ):
        rating_meta = schema.get("review_rating", {})
        p80 = rating_meta.get("p80") if isinstance(rating_meta, dict) else None
        if p80 is not None:
            filters["review_rating"] = {"$gte": p80}

    return filters


class SchemaIntentResolver:
    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        self.schema = schema or {}

    def resolve(self, user_message: str) -> Optional[Dict[str, Any]]:
        text = (user_message or "").strip()
        if not text:
            return None

        lowered = text.lower()
        if any(k in lowered for k in ("import", "load dataset", "ingest")):
            return None

        is_count = any(h in lowered for h in _COUNT_HINTS)
        is_grouped = any(h in lowered for h in _GROUP_HINTS)
        is_top = any(h in lowered for h in _TOP_HINTS)
        is_bottom = any(h in lowered for h in _BOTTOM_HINTS)
        is_rated = any(re.search(rf"\b{re.escape(h)}\b", lowered) for h in _RATED_HINTS)
        is_avg = any(h in lowered for h in _AVG_HINTS)
        is_min = any(h in lowered for h in _MIN_HINTS)
        is_max = any(h in lowered for h in _MAX_HINTS)
        mentions_products = any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in _PRODUCT_HINTS)
        wants_rows = any(w in lowered for w in ("show", "list", "find", "display"))

        limit = _parse_limit(text, default=10)
        filters = _extract_filters(text, self.schema)
        metric = _detect_metric_field(text, self.schema)
        group_by = _extract_group_by(text, self.schema)

        if is_count and not is_grouped and not is_avg:
            return {
                "tool": "query_shopping_trends",
                "params": {"filter": filters, "count_only": True},
            }

        if is_avg:
            agg_field = metric or ("review_rating" if is_rated else "purchase_amount_usd")
            grp = group_by or "category"
            return {
                "tool": "summarize_shopping_trends",
                "params": {
                    "group_by": grp,
                    "aggregations": [
                        {"op": "avg", "field": agg_field, "alias": f"avg_{agg_field}"},
                        {"op": "count", "alias": "total_purchases"},
                    ],
                    "sort_by": f"avg_{agg_field}",
                    "sort_order": "asc" if is_bottom else "desc",
                    "limit": limit,
                    "filter": filters,
                },
            }

        if is_grouped and (is_top or is_count or "most common" in lowered or "popular" in lowered):
            grp = group_by or ("category" if "categor" in lowered else "item_purchased")
            return {
                "tool": "summarize_shopping_trends",
                "params": {
                    "group_by": grp,
                    "aggregations": [{"op": "count", "alias": "total_purchases"}],
                    "sort_by": "total_purchases",
                    "sort_order": "asc" if is_bottom else "desc",
                    "limit": limit,
                    "filter": filters,
                },
            }

        if is_min or is_max:
            target = metric or "purchase_amount_usd"
            return {
                "tool": "query_shopping_trends",
                "params": {
                    "filter": filters,
                    "sort_by": target,
                    "sort_order": "desc" if is_max else "asc",
                    "limit": limit,
                },
            }

        if is_top or is_bottom or is_rated or mentions_products or wants_rows or filters:
            sort_by = (
                "review_rating"
                if (is_rated or "top rated" in lowered or "best" in lowered)
                else (metric or "purchase_amount_usd")
            )
            sort_order = "asc" if (is_bottom or "cheapest" in lowered or "worst" in lowered) else "desc"
            if "latest" in lowered or "newest" in lowered:
                sort_by = "customer_id"
                sort_order = "desc"
            if "cheapest" in lowered or "cheap" in lowered:
                sort_by = "purchase_amount_usd"
                sort_order = "asc"
            if "expensive" in lowered or "highest price" in lowered:
                sort_by = "purchase_amount_usd"
                sort_order = "desc"
            return {
                "tool": "query_shopping_trends",
                "params": {
                    "filter": filters,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                    "limit": limit,
                },
            }

        return {"tool": "query_shopping_trends", "params": {"filter": {}, "limit": 10}}
