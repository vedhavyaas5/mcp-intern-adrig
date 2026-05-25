from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required server file: {path}\n"
            "Create it first (or update cockpit_api.py to point at the correct path)."
        )


def _get_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        return ChatGroq(model=model, temperature=0.1, groq_api_key=groq_key)

    # Local fallback LLM for development when GROQ_API_KEY is not set.
    # This implements minimal, deterministic behaviors used by the router
    # and shopping-trends helper prompts so the server can run offline.
    class _DummyResponse:
        def __init__(self, content: str):
            self.content = content

    class _DummyLLM:
        async def ainvoke(self, messages: list[Any]):
            # Extract system/human message content
            system_text = ""
            human_text = ""
            for m in messages:
                try:
                    c = getattr(m, "content", str(m))
                except Exception:
                    c = str(m)
                if isinstance(m, SystemMessage):
                    system_text = str(c or "")
                elif isinstance(m, HumanMessage):
                    human_text = str(c or "")
                else:
                    # fallback: treat last as human
                    human_text = str(c or "")

            # Simple router heuristics
            if "You are a JSON router" in system_text:
                txt = (human_text or "").strip()
                # ticket id
                match = re.search(r"\bTKT-\d+\b", txt, flags=re.IGNORECASE)
                if match:
                    return _DummyResponse(json.dumps({"kind": "tool", "tool_name": "read_ticket_record", "tool_args": {"ticket_id": match.group(0)}}))
                if any(k in txt.lower() for k in ("import", "load", "ingest")):
                    return _DummyResponse(json.dumps({"kind": "tool", "tool_name": "import_shopping_trends_csv_to_mongodb", "tool_args": {}}))
                if txt.isdigit():
                    return _DummyResponse(json.dumps({"kind": "tool", "tool_name": "get_shopping_trend_by_customer_id", "tool_args": {"customer_id": int(txt)}}))
                return _DummyResponse(json.dumps({"kind": "answer", "answer": "Local fallback: GROQ API key not set. For full LLM features set GROQ_API_KEY. Ask 'import' to load the dataset into MongoDB."}))

            # Shopping trends tool prompt heuristics
            if "You decide how to answer a question about a shopping trends dataset" in system_text:
                txt = (human_text or "").lower()
                for field in _SHOPPING_TRENDS_FIELDS:
                    if field in txt:
                        if any(k in txt for k in ("top", "most", "summary", "summarize", "count", "breakdown")):
                            return _DummyResponse(json.dumps({"tool_name": "summarize_shopping_trends", "tool_args": {"field": field, "limit": 10}}))
                # default: return a generic query_shopping_trends with empty filter
                return _DummyResponse(json.dumps({"tool_name": "query_shopping_trends", "tool_args": {"query_json": {}, "limit": 10}}))

            # Default: echo empty content
            return _DummyResponse("")

    return _DummyLLM()


AllowedToolName = Literal[
    "read_ticket_record",
    "escalate_to_human_queue",
    "query_product_catalog",
    "search_elasticsearch_catalog",
    "generate_checkout_incentive",
    "import_shopping_trends_csv_to_mongodb",
    "get_shopping_trend_by_customer_id",
    "summarize_shopping_trends",
    "query_shopping_trends",
]


_ALLOWED_TOOLS: set[str] = {
    "read_ticket_record",
    "escalate_to_human_queue",
    "query_product_catalog",
    "search_elasticsearch_catalog",
    "generate_checkout_incentive",
    "import_shopping_trends_csv_to_mongodb",
    "get_shopping_trend_by_customer_id",
    "summarize_shopping_trends",
    "query_shopping_trends",
}


_ROUTER_SYSTEM_PROMPT = """
You are a JSON router for a backend that can call MCP tools.

Return JSON ONLY (no markdown, no commentary). Output must be one of:

1) Tool call:
{
    "kind": "tool",
    "tool_name": "<one of the allowed tools>",
    "tool_args": {"<arg>": <value>}
}

2) Direct answer (no tool):
{
    "kind": "answer",
    "answer": "<short helpful answer>"
}

Allowed tools + args:
- read_ticket_record(ticket_id: string like "TKT-101")
- escalate_to_human_queue(ticket_id: string, reason: string)
- query_product_catalog(search_intent: string)
- search_elasticsearch_catalog(search_intent: string)
- generate_checkout_incentive(context: string optional)
- import_shopping_trends_csv_to_mongodb(csv_path: string optional)
- get_shopping_trend_by_customer_id(customer_id: integer)
- summarize_shopping_trends(field: string, limit: integer optional)
- query_shopping_trends(query_json: string, limit: integer optional, count_only: boolean optional)

Routing rules:
- If text includes a ticket id (TKT-<number>), use read_ticket_record.
- If the user expresses extreme frustration/financial loss and includes a ticket id, use escalate_to_human_queue with a short reason.
- For company product/pricing/features, use query_product_catalog or search_elasticsearch_catalog.
    Company products are the internal offerings (e.g. Nexus Traffic-Vision Monitoring Suite).
- For discount/checkout/code requests, use generate_checkout_incentive.
- For shopping trends / dataset questions:
    - If user asks to "import" or "load" the dataset, call import_shopping_trends_csv_to_mongodb.
    - If a standalone integer is present, treat it as customer_id and call get_shopping_trend_by_customer_id.
    - If user asks for a summary like "top categories" or "count by location", call summarize_shopping_trends.
    - For any other query, search, filtering, or question about items/colors/categories/seasons/customers in the shopping trends dataset, call query_shopping_trends with a MongoDB query JSON (e.g. {"color": "White"}).
    - If the user asks for colored items/products, sizes, seasons, locations, genders, payment methods, or similar attributes, this is shopping trends (not the company catalog).
    - If the user asks for a count/number of records (e.g. "how many", "count", "number of"), use query_shopping_trends with count_only=true.

Constraints:
- Choose exactly ONE tool call at most.
- If using summarize_shopping_trends or query_shopping_trends, cap limit to <= 50.
- If you cannot map to a tool, return kind=answer.
""".strip()

_SHOPPING_TRENDS_FIELDS = {
        "customer_id",
        "age",
        "gender",
        "item_purchased",
        "category",
        "purchase_amount_usd",
        "location",
        "size",
        "color",
        "season",
        "review_rating",
        "subscription_status",
        "payment_method",
        "shipping_type",
        "discount_applied",
        "promo_code_used",
        "previous_purchases",
        "preferred_payment_method",
        "frequency_of_purchases",
}

_SHOPPING_TRENDS_HINTS = {
        "color",
        "colour",
        "colored",
        "coloured",
        "season",
        "category",
        "item",
        "items",
        "size",
        "gender",
        "location",
        "payment",
        "shipping",
        "subscription",
        "promo",
        "discount",
        "review",
        "rating",
        "purchase",
        "customer",
        "age",
        "frequency",
}

_SHOPPING_TRENDS_SUMMARY_HINTS = {
        "top",
        "count",
        "summary",
        "summarize",
        "breakdown",
        "distribution",
        "group",
        "most",
        "least",
        "popular",
        "trend",
}

_SHOPPING_TRENDS_COLOR_WORDS = {
        "white",
        "black",
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "grey",
        "beige",
        "cream",
        "gold",
        "silver",
}

_FIELD_ALIASES = {
        "customerid": "customer_id",
        "customer_id": "customer_id",
        "customer id": "customer_id",
        "item purchased": "item_purchased",
        "item_purchased": "item_purchased",
        "purchase amount": "purchase_amount_usd",
        "purchase_amount": "purchase_amount_usd",
        "purchase_amount_usd": "purchase_amount_usd",
        "review rating": "review_rating",
        "review_rating": "review_rating",
        "subscription status": "subscription_status",
        "subscription_status": "subscription_status",
        "payment method": "payment_method",
        "payment_method": "payment_method",
        "shipping type": "shipping_type",
        "shipping_type": "shipping_type",
        "discount applied": "discount_applied",
        "discount_applied": "discount_applied",
        "promo code used": "promo_code_used",
        "promo_code_used": "promo_code_used",
        "previous purchases": "previous_purchases",
        "previous_purchases": "previous_purchases",
        "preferred payment method": "preferred_payment_method",
        "preferred_payment_method": "preferred_payment_method",
        "frequency of purchases": "frequency_of_purchases",
        "frequency_of_purchases": "frequency_of_purchases",
    }

_PRODUCT_CATALOG_HINTS = {
    "catalog",
    "pricing",
    "price",
    "feature",
    "features",
    "sku",
    "quote",
    "demo",
    "plan",
}

_SHOPPING_TRENDS_TOOL_PROMPT = """
You decide how to answer a question about a shopping trends dataset.

Return JSON ONLY with this shape:
{
        "tool_name": "summarize_shopping_trends" | "query_shopping_trends",
        "tool_args": { ... }
}

Rules:
- Use only these fields: customer_id, age, gender, item_purchased, category, purchase_amount_usd,
    location, size, color, season, review_rating, subscription_status, payment_method, shipping_type,
    discount_applied, promo_code_used, previous_purchases, preferred_payment_method, frequency_of_purchases.
- Field names must use underscores (e.g. shipping_type, payment_method).
- If the user asks for a summary/breakdown/top/most/least counts by a field, use summarize_shopping_trends
    with tool_args {"field": "<field>", "limit": <integer optional>}.
- If the user asks for a count/number of records matching conditions, use query_shopping_trends
    with tool_args {"query_json": <object or JSON string>, "count_only": true}.
- Otherwise use query_shopping_trends with tool_args {"query_json": <object or JSON string>, "limit": <integer optional>}.
- Default to equality filters.
- Use $gt/$lt/$gte/$lte only if the user asks for ranges.
- If the user asks for multiple values, use $in.
- If unsure, return query_shopping_trends with an empty filter: {"query_json": {}}.
""".strip()


def _extract_latest_user_text(messages: list[tuple[str, str]]) -> str:
    for role, content in reversed(messages):
        if role == "user" and content:
            return content
    # Fallback: return last message content.
    return messages[-1][1] if messages else ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parsing for a JSON object from an LLM response."""
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Fallback: extract the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return None


def _normalize_field_name(raw: str) -> str:
    cleaned = (raw or "").strip().lower()
    if not cleaned:
        return ""
    if cleaned in _FIELD_ALIASES:
        return _FIELD_ALIASES[cleaned]
    cleaned = cleaned.replace(" ", "_")
    return _FIELD_ALIASES.get(cleaned, cleaned)


def _looks_like_shopping_trends_query(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False

    if re.search(r"\bTKT-\d+\b", text, flags=re.IGNORECASE):
        return False

    if any(hint in lowered for hint in {"import", "load", "ingest"}):
        return False

    if any(hint in lowered for hint in _PRODUCT_CATALOG_HINTS):
        if not any(hint in lowered for hint in _SHOPPING_TRENDS_HINTS):
            if not any(color in lowered for color in _SHOPPING_TRENDS_COLOR_WORDS):
                return False

    if any(field in lowered for field in _SHOPPING_TRENDS_FIELDS):
        return True

    if re.search(r"\bcustomer\s*id\b", lowered):
        return True

    if any(hint in lowered for hint in _SHOPPING_TRENDS_HINTS):
        return True

    if any(color in lowered for color in _SHOPPING_TRENDS_COLOR_WORDS):
        if any(token in lowered for token in {"product", "products", "item", "items", "color", "colour"}):
            return True

    return False


def _coerce_query_json(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, str):
        return value.strip() or "{}"
    return "{}"


async def _build_shopping_trends_query(llm: Any, user_text: str) -> dict[str, Any] | None:
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=_SHOPPING_TRENDS_TOOL_PROMPT),
                HumanMessage(content=user_text),
            ]
        )
    except Exception:
        return None

    parsed = _parse_json_object(str(getattr(response, "content", "") or ""))
    if not parsed:
        return None

    tool_name = str(parsed.get("tool_name") or "").strip()
    tool_args = parsed.get("tool_args")
    if not isinstance(tool_args, dict):
        tool_args = {}

    if tool_name == "summarize_shopping_trends":
        return {
            "kind": "tool",
            "tool_name": tool_name,
            "tool_args": tool_args,
        }

    if tool_name == "query_shopping_trends":
        query_json = _coerce_query_json(tool_args.get("query_json"))
        limit = tool_args.get("limit", 10)
        count_only = _parse_bool(tool_args.get("count_only"))
        clean_args: dict[str, Any] = {"query_json": query_json, "limit": limit}
        if count_only is not None:
            clean_args["count_only"] = count_only
        return {
            "kind": "tool",
            "tool_name": tool_name,
            "tool_args": clean_args,
        }

    return None


def _tool_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(json.dumps(item, ensure_ascii=True))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=True)
    return str(content)


def _friendly_tool_error(tool_name: str, tool_text: str, user_text: str) -> str:
    # Keep this user-facing and avoid leaking stack traces / connection strings.
    if tool_name in {"read_ticket_record", "escalate_to_human_queue"}:
        ticket = None
        match = re.search(r"\bTKT-\d+\b", user_text, flags=re.IGNORECASE)
        if match:
            ticket = match.group(0).upper()
        if "does not exist" in tool_text or "No record" in tool_text:
            return f"I couldn't find that ticket{f' ({ticket})' if ticket else ''}. Please double-check the ticket id."
        return "I couldn't complete the ticket action right now. Please try again."

    if "MongoDB" in tool_text or "import" in tool_text.lower():
        return (
            "I couldn't reach the shopping trends database. "
            "Make sure MongoDB is running and the dataset is imported, then try again."
        )

    if "Invalid field" in tool_text or "Invalid field name" in tool_text:
        return "That field name isn't valid. Try fields like category, location, season, gender, or payment_method."

    return "I couldn't complete that request right now. Please try again."


async def _adjust_spelling_and_optimize_query(llm: Any, user_text: str) -> str:
    prompt = """
You are a helpful assistant. Correct any spelling errors, typos, grammatical mistakes, or regional spelling variations (like "coloured" -> "colored", "TKT101" -> "TKT-101", etc.) in the user's input.
Standardize common dataset terms where appropriate (e.g. color, item, category, season) without changing intent.
Maintain the exact original intent and meaning. If the text has no errors, return it exactly as is.
Return only the corrected/standardized text, with no introductory text, explanation, or commentary.
""".strip()
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=user_text),
            ]
        )
        content = str(getattr(response, "content", "") or "").strip()
        if content:
            return content
    except Exception:
        pass
    return user_text


async def _route_with_llm(llm: Any, user_text: str) -> dict[str, Any]:
    if _looks_like_shopping_trends_query(user_text):
        decision = await _build_shopping_trends_query(llm, user_text)
        if decision:
            return decision

    response = await llm.ainvoke(
        [
            SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ]
    )
    parsed = _parse_json_object(str(getattr(response, "content", "") or ""))
    return parsed or {"kind": "answer", "answer": "I couldn't understand that. Could you rephrase?"}


def _sanitize_tool_args(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = dict(tool_args or {})

    if tool_name == "read_ticket_record":
        ticket_id = str(args.get("ticket_id") or "").strip()
        if not ticket_id:
            raise ValueError("missing ticket_id")
        return {"ticket_id": ticket_id}

    if tool_name == "escalate_to_human_queue":
        ticket_id = str(args.get("ticket_id") or "").strip()
        reason = str(args.get("reason") or "").strip() or "User requested escalation"
        if not ticket_id:
            raise ValueError("missing ticket_id")
        return {"ticket_id": ticket_id, "reason": reason}

    if tool_name in {"query_product_catalog", "search_elasticsearch_catalog"}:
        search_intent = str(args.get("search_intent") or "").strip()
        if not search_intent:
            raise ValueError("missing search_intent")
        return {"search_intent": search_intent}

    if tool_name == "generate_checkout_incentive":
        context = str(args.get("context") or "").strip()
        return {"context": context}

    if tool_name == "import_shopping_trends_csv_to_mongodb":
        csv_path = str(args.get("csv_path") or "").strip()
        return {"csv_path": csv_path} if csv_path else {}

    if tool_name == "get_shopping_trend_by_customer_id":
        cid = args.get("customer_id")
        try:
            return {"customer_id": int(cid)}
        except Exception as exc:
            raise ValueError("invalid customer_id") from exc

    if tool_name == "summarize_shopping_trends":
        field_raw = str(args.get("field") or "").strip()
        field = _normalize_field_name(field_raw)
        if not field:
            raise ValueError("missing field")
        limit_raw = args.get("limit", 10)
        try:
            limit = int(limit_raw)
        except Exception:
            limit = 10
        limit = max(1, min(limit, 50))
        return {"field": field, "limit": limit}

    if tool_name == "query_shopping_trends":
        query_json = str(args.get("query_json") or "{}").strip()
        limit_raw = args.get("limit", 10)
        count_only_raw = args.get("count_only")
        try:
            limit = int(limit_raw)
        except Exception:
            limit = 10
        limit = max(1, min(limit, 50))
        count_only = _parse_bool(count_only_raw)
        clean_args = {"query_json": query_json, "limit": limit}
        if count_only is not None:
            clean_args["count_only"] = count_only
        return clean_args

    return args


async def _rewrite_with_llm(llm: Any, user_text: str, tool_text: str) -> str:
    writer_prompt = """
You are a helpful assistant. You will be given a user's request and the result from a backend tool call.

Write a detailed, user-friendly answer based ONLY on the tool result. Do not mention tools, MCP, routing, or internal systems.
If the tool result indicates no matches, explain that clearly and suggest what the user can try next.
""".strip()

    response = await llm.ainvoke(
        [
            SystemMessage(content=writer_prompt),
            HumanMessage(
                content=(
                    f"User request:\n{user_text}\n\n"
                    f"Tool result:\n{tool_text}\n"
                )
            ),
        ]
    )
    return str(getattr(response, "content", "") or "").strip() or tool_text


async def _chat(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    message = str(payload.get("message") or "").strip()
    messages_payload = payload.get("messages")

    if not message and not messages_payload:
        return JSONResponse({"error": "Missing 'message'"}, status_code=400)

    # Accept either a single message, or a simple messages[] history.
    messages: list[tuple[str, str]] = []
    if isinstance(messages_payload, list):
        for item in messages_payload:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant", "system"} and isinstance(content, str) and content:
                messages.append((role, content))

    if not messages:
        messages = [("user", message)]

    user_text = _extract_latest_user_text(messages)

    llm = request.app.state.llm
    tools_by_name: dict[str, BaseTool] = request.app.state.tools_by_name
    mcp_lock: asyncio.Lock = request.app.state.mcp_lock

    # Pre-process: adjust spelling and optimize the query using the LLM
    user_text = await _adjust_spelling_and_optimize_query(llm, user_text)

    # 1) Use the LLM only to understand intent + pick one tool.
    try:
        decision = await _route_with_llm(llm, user_text)
    except Exception:
        # Keep the endpoint stable; return a friendly message.
        return JSONResponse({"reply": "I couldn't process that right now. Please try again."})

    kind = str(decision.get("kind") or "").strip().lower()
    if kind == "answer":
        answer = str(decision.get("answer") or "").strip()
        if not answer:
            answer = "I couldn't understand that. Could you rephrase?"
        return JSONResponse({"reply": answer})

    if kind != "tool":
        return JSONResponse({"reply": "I couldn't understand that. Could you rephrase?"})

    tool_name = str(decision.get("tool_name") or "").strip()
    if tool_name not in _ALLOWED_TOOLS:
        return JSONResponse({"reply": "I couldn't map that request to an available action. Please rephrase."})

    tool_args_raw = decision.get("tool_args")
    if tool_args_raw is None:
        tool_args_raw = {}
    if not isinstance(tool_args_raw, dict):
        return JSONResponse({"reply": "I couldn't read the request details. Please rephrase."})

    # 2) Call exactly one MCP tool. Tool sessions are not assumed concurrency-safe.
    tool = tools_by_name.get(tool_name)
    if tool is None:
        return JSONResponse({"reply": "That action is not available right now. Please try again."})

    try:
        tool_args = _sanitize_tool_args(tool_name, tool_args_raw)
    except Exception:
        return JSONResponse({"reply": "I need a bit more detail to do that. Please rephrase your request."})

    try:
        async with mcp_lock:
            tool_result = await tool.ainvoke(tool_args)
    except Exception:
        return JSONResponse({"reply": "I couldn't complete that request right now. Please try again."})

    # Normalize tool result into text.
    tool_content: Any = tool_result
    if isinstance(tool_result, dict) and "content" in tool_result:
        tool_content = tool_result.get("content")
    elif isinstance(tool_result, tuple) and len(tool_result) == 2:
        tool_content = tool_result[0]

    tool_text = _tool_content_to_text(tool_content).strip()
    if not tool_text:
        tool_text = "(no result)"

    if tool_text.startswith("❌"):
        return JSONResponse({"reply": _friendly_tool_error(tool_name, tool_text, user_text)})

    # 3) Optionally rewrite tool output into a user-friendly answer.
    reply_mode = str(os.getenv("LLM_REPLY_MODE", "rewrite")).strip().lower()
    if reply_mode == "raw":
        return JSONResponse({"reply": tool_text})

    try:
        rewritten = await _rewrite_with_llm(llm, user_text, tool_text)
        return JSONResponse({"reply": rewritten})
    except Exception:
        return JSONResponse({"reply": tool_text})


async def _health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


@asynccontextmanager
async def _lifespan(app: Starlette):
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=project_root / ".env")

    src_dir = project_root / "src"
    support_server = src_dir / "support_server.py"
    sales_server = src_dir / "sales_server.py"

    _require_file(support_server)
    _require_file(sales_server)

    connections = {
        "support_system": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-u", str(support_server)],
            "cwd": str(project_root),
        },
        "sales_system": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-u", str(sales_server)],
            "cwd": str(project_root),
        },
    }

    stack = AsyncExitStack()

    mcp_bridge = MultiServerMCPClient(connections)
    support_session = await stack.enter_async_context(mcp_bridge.session("support_system"))
    sales_session = await stack.enter_async_context(mcp_bridge.session("sales_system"))

    support_tools = await load_mcp_tools(support_session, server_name="support_system")
    sales_tools = await load_mcp_tools(sales_session, server_name="sales_system")
    tools = [*support_tools, *sales_tools]

    llm = _get_llm()

    # Store tools for direct invocation (LLM decides which tool + args to call).
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools if getattr(t, "name", None)}

    # Enforce a whitelist in case MCP servers expose additional tools in the future.
    tools_by_name = {name: tool for name, tool in tools_by_name.items() if name in _ALLOWED_TOOLS}

    app.state.llm = llm
    app.state.tools_by_name = tools_by_name
    app.state.mcp_lock = asyncio.Lock()

    try:
        yield
    finally:
        await stack.aclose()


app = Starlette(
    debug=True,
    lifespan=_lifespan,
    routes=[
        Route("/api/health", _health, methods=["GET"]),
        Route("/api/chat", _chat, methods=["POST"]),
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
