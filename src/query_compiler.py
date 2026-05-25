from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Canonical semantic aliases -> MongoDB field (shared with intent_resolver)
FIELD_ALIASES: Dict[str, str] = {
    "rating": "review_rating",
    "rated": "review_rating",
    "review": "review_rating",
    "reviews": "review_rating",
    "star": "review_rating",
    "stars": "review_rating",
    "score": "review_rating",
    "price": "purchase_amount_usd",
    "cost": "purchase_amount_usd",
    "amount": "purchase_amount_usd",
    "expensive": "purchase_amount_usd",
    "cheap": "purchase_amount_usd",
    "type": "category",
    "group": "category",
    "section": "category",
    "product": "item_purchased",
    "products": "item_purchased",
    "item": "item_purchased",
    "items": "item_purchased",
    "popular": "item_purchased",
    "selling": "item_purchased",
    "shoes": "item_purchased",
    "jackets": "item_purchased",
    "jacket": "item_purchased",
    "accessories": "category",
    "footwear": "category",
    "outerwear": "category",
    "clothing": "category",
}

COMPILER_HEADER = """\
You are an advanced Schema-Aware MCP Query Compiler.

Your ONLY responsibility is:
Convert natural language user requests into structured MCP query JSON.

You are NOT a chatbot.
You do NOT explain answers.
You do NOT generate conversational text.
You do NOT apologize.
You do NOT hallucinate missing data.

You ONLY generate valid JSON.

## OBJECTIVE

Given database schema summary, column semantics, statistical metadata (p20/p80), and the user request:
- understand intent
- map semantics to database fields
- infer sorting / filtering / grouping / counting
- generate optimized MCP query JSON

## DATABASE UNDERSTANDING

top rated / best products -> query_shopping_trends sorted by review_rating desc (optionally with p80 threshold)
worst / lowest rated -> query_shopping_trends sorted by review_rating asc
most expensive / highest price -> query_shopping_trends sorted by purchase_amount_usd desc
cheapest / lowest price -> query_shopping_trends sorted by purchase_amount_usd asc
popular / best selling / most purchased -> summarize_shopping_trends using count desc by item/category
how many / count -> query_shopping_trends with count_only true

## SEMANTIC COLUMN MATCHING

Match user words to schema columns using aliases (rating, review, stars, price, cost, category, season, color, etc.).
Prefer exact field names from schema. Use underscores (review_rating, item_purchased, purchase_amount_usd).

## STRICT OUTPUT RULES

- Return JSON ONLY
- NEVER markdown, prose, apologies, or "I couldn't find"
- Single JSON object: {"tool":"...","params":{...}}

## MCP TOOLS (pick one)

1) query_shopping_trends — row-level fetch, filter, sort, count
   params: filter (dict), sort_by, sort or sort_order (asc|desc), limit, count_only (bool)

2) summarize_shopping_trends — group-by rankings, averages, popularity, breakdowns
   params: group_by, aggregations [{op, field, alias}], sort_by, sort or sort_order, limit, filter (dict)

## SORTING

top / best / highest / max -> desc
lowest / cheapest / worst / min -> asc

## FILTERS (Mongo operators in filter dict)

under X -> {"$lte": X}
above X -> {"$gte": X}
between A and B -> {"$gte": A, "$lte": B}

## AGGREGATION (summarize tool)

how many / count -> op count
average -> op avg
maximum -> op max
minimum -> op min

## PERCENTILE INTELLIGENCE

Use p80/p20 ONLY for query_shopping_trends row filters, NOT before summarize group-by averages.

## TOOL CHOICE

- List/show/find rows with filters -> query_shopping_trends
- how many matching -> query_shopping_trends + count_only true
- top N categories / breakdown / per field / average per group -> summarize_shopping_trends
- most common color / top categories -> summarize_shopping_trends with count

## EXAMPLES (JSON only)

User: top rated winter jackets under 100
{"tool":"query_shopping_trends","params":{"filter":{"season":"Winter","purchase_amount_usd":{"$lte":100}},"sort_by":"review_rating","sort_order":"desc","limit":10}}

User: how many white shoes
{"tool":"query_shopping_trends","params":{"filter":{"color":"White","category":"Footwear"},"aggregate":"count"}}

User: cheapest accessories
{"tool":"query_shopping_trends","params":{"filter":{"category":"Accessories"},"sort_by":"purchase_amount_usd","sort_order":"asc","limit":10}}

User: top rated products
{"tool":"query_shopping_trends","params":{"sort_by":"review_rating","sort_order":"desc","limit":10}}

User: suggest me best item
{"tool":"query_shopping_trends","params":{"sort_by":"review_rating","sort_order":"desc","limit":10}}

User: top 5 categories by revenue
{"tool":"summarize_shopping_trends","params":{"group_by":"category","aggregations":[{"op":"sum","field":"purchase_amount_usd","alias":"revenue"},{"op":"count","alias":"total_purchases"}],"sort_by":"revenue","sort_order":"desc","limit":5}}

Natural Language -> MCP Query JSON. NOT chat responses.
"""


def build_compiler_prompt(schema_text: str) -> str:
    return f"{COMPILER_HEADER}\n\n## SCHEMA\n{schema_text}\n"


def _coerce_sort(params: Dict[str, Any]) -> None:
    if "sort_order" in params and "sort" not in params:
        params["sort"] = str(params.pop("sort_order")).strip().lower()
    sort_val = params.get("sort")
    if isinstance(sort_val, str):
        params["sort"] = sort_val.lower()


def _aggregate_to_count_only(params: Dict[str, Any]) -> bool:
    agg = params.get("aggregate")
    if agg is None:
        return False
    if isinstance(agg, str) and agg.strip().lower() in {"count", "how_many", "total"}:
        params["count_only"] = True
        params.pop("aggregate", None)
        return True
    return False


def _upgrade_query_to_summarize(params: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert query_shopping_trends params with group_by into summarize_shopping_trends."""
    group_by = str(params.pop("group_by", "") or "").strip()
    flt = params.pop("filter", None) or params.pop("query_json", None)
    if isinstance(flt, str):
        try:
            flt = json.loads(flt)
        except json.JSONDecodeError:
            flt = {}
    if not isinstance(flt, dict):
        flt = {}

    sort_by = str(params.get("sort_by") or "total_purchases").strip()
    _coerce_sort(params)
    sort_dir = params.get("sort", "desc")
    limit = params.get("limit", 10)

    aggregations: List[Dict[str, Any]] = list(params.pop("aggregations", None) or [])
    agg_field = str(params.pop("aggregate_field", "") or "").strip()
    agg_op = str(params.pop("aggregate", "") or "").strip().lower()

    if not aggregations:
        if agg_op in {"avg", "average"} and agg_field:
            aggregations = [
                {"op": "avg", "field": agg_field, "alias": f"avg_{agg_field}"},
                {"op": "count", "alias": "total_purchases"},
            ]
            sort_by = f"avg_{agg_field}"
        elif agg_op in {"sum"} and agg_field:
            aggregations = [
                {"op": "sum", "field": agg_field, "alias": "revenue"},
                {"op": "count", "alias": "total_purchases"},
            ]
            sort_by = "revenue"
        elif agg_op in {"max", "maximum"} and agg_field:
            aggregations = [
                {"op": "max", "field": agg_field, "alias": f"max_{agg_field}"},
                {"op": "count", "alias": "total_purchases"},
            ]
            sort_by = f"max_{agg_field}"
        elif agg_op in {"min", "minimum"} and agg_field:
            aggregations = [
                {"op": "min", "field": agg_field, "alias": f"min_{agg_field}"},
                {"op": "count", "alias": "total_purchases"},
            ]
            sort_by = f"min_{agg_field}"
        else:
            aggregations = [
                {"op": "count", "alias": "total_purchases"},
                {"op": "sum", "field": "purchase_amount_usd", "alias": "revenue"},
            ]
            if sort_by in {"", "review_rating"} or "rating" in sort_by:
                aggregations.insert(
                    0, {"op": "avg", "field": "review_rating", "alias": "avg_rating"}
                )
                sort_by = "avg_rating"

    out: Dict[str, Any] = {
        "group_by": group_by or "category",
        "aggregations": aggregations,
        "sort_by": sort_by,
        "sort": sort_dir,
        "limit": limit,
    }
    if flt:
        out["filter"] = flt
    return out


def normalize_compiler_output(
    tool_call: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize compiler JSON into MCP-ready tool + params.
    Maps sort_order, filter, aggregate aliases; upgrades grouped queries to summarize.
    """
    tool = str(tool_call.get("tool") or tool_call.get("tool_name") or "").strip()
    params = tool_call.get("params")
    if params is None:
        params = tool_call.get("tool_args")
    if not isinstance(params, dict):
        params = {}

    params = dict(params)
    _coerce_sort(params)

    if tool == "query_shopping_trends":
        _aggregate_to_count_only(params)

        if params.get("group_by") or params.get("aggregations"):
            sum_params = _upgrade_query_to_summarize(params, schema)
            return {"tool": "summarize_shopping_trends", "params": sum_params}

        if "filter" in params and "query_json" not in params:
            params["query_json"] = params.pop("filter")

        # Top-rated rows: apply p80 only when explicitly sorting by rating (not plain counts)
        if (
            schema
            and not params.get("count_only")
            and params.get("sort_by") == "review_rating"
            and str(params.get("sort", "desc")).lower() == "desc"
        ):
            q = params.get("query_json")
            if isinstance(q, dict) and "review_rating" not in q:
                meta = schema.get("review_rating", {})
                if meta.get("type") == "number" and meta.get("p80") is not None:
                    q = dict(q)
                    q["review_rating"] = {"$gte": meta["p80"]}
                    params["query_json"] = q

    elif tool == "summarize_shopping_trends":
        if "filter" in params and "query_json" not in params:
            pass  # summarize uses filter dict directly
        if not params.get("aggregations"):
            agg_op = str(params.get("aggregate", "") or "").lower()
            agg_field = str(params.get("aggregate_field", "") or "").strip()
            if agg_op == "count":
                params["aggregations"] = [{"op": "count", "alias": "total_purchases"}]
                params.setdefault("sort_by", "total_purchases")
            elif agg_op in {"avg", "average"} and agg_field:
                params["aggregations"] = [
                    {"op": "avg", "field": agg_field, "alias": f"avg_{agg_field}"},
                    {"op": "count", "alias": "total_purchases"},
                ]
                params.setdefault("sort_by", f"avg_{agg_field}")

    return {"tool": tool, "params": params}



def apply_safety_rules(
    tool: str,
    params: Dict[str, Any],
    user_message: str,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Correct common LLM mistakes before MCP execution without overriding valid user intent.
    """
    params = dict(params or {})
    _coerce_sort(params)

    if tool == "summarize_shopping_trends":
        group_by = str(params.get("group_by") or "").strip()
        if group_by == "item_purchased":
            flt = params.get("filter")
            if isinstance(flt, dict) and "review_rating" in flt:
                rr = flt.get("review_rating")
                if isinstance(rr, dict) and "$gte" in rr and len(rr) == 1:
                    flt = dict(flt)
                    flt.pop("review_rating", None)
                    if flt:
                        params["filter"] = flt
                    else:
                        params.pop("filter", None)

    return {"tool": tool, "params": params}


def strip_non_json_llm_output(raw: str) -> str:
    """Keep only the first JSON object from model output."""
    cleaned = re.sub(r"```(?:json)?", "", raw or "").strip().rstrip("`").strip()
    if cleaned.startswith("{"):
        return cleaned
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return match.group() if match else cleaned
