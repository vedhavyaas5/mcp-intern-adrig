from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.intent_resolver import SchemaIntentResolver
from src.query_compiler import (
    apply_safety_rules,
    build_compiler_prompt,
    normalize_compiler_output,
    strip_non_json_llm_output,
)


def parse_mcp_payload(raw: Any) -> Any:
    """Normalize MCP tool output to Python structures for formatting."""
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("❌") or text.startswith("⚠️"):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "Counted " in text and "matching records" in text:
        m = re.search(r"Counted\s+(\d+)", text)
        if m:
            return {"count": int(m.group(1))}
    return text


def build_system_prompt(schema_text: str) -> str:
    return build_compiler_prompt(schema_text)


VALID_TOOLS = {"query_shopping_trends", "summarize_shopping_trends"}


def validate_tool_call(data: Any) -> Tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Output is not a JSON object"
    if "tool" not in data:
        return False, "Missing 'tool' key"
    if data.get("tool") not in VALID_TOOLS:
        return False, f"Unknown tool '{data.get('tool')}'. Must be one of {sorted(VALID_TOOLS)}"
    if "params" not in data:
        return False, "Missing 'params' key"
    if not isinstance(data.get("params"), dict):
        return False, "'params' must be a JSON object"
    return True, ""


def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_non_json_llm_output(raw_text or "")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    return None


def _normalize_tool_call(
    tool_call: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return normalize_compiler_output(tool_call, schema=schema)


def _humanize(key: str) -> str:
    return key.replace("_", " ").title()

def _is_greeting(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    return lowered in {
        "hi",
        "hello",
        "hey",
        "yo",
        "hola",
        "good morning",
        "good afternoon",
        "good evening",
    }


def _greeting_reply() -> str:
    return (
        "Hi! Ask about shopping trends (for example: top rated products, top 5 categories, "
        "or how many white shoes), or support tickets (for example: TKT-101)."
    )


class ChatbotFramework:
    def __init__(
        self,
        schema_text: str,
        llm: Any,
        tool_caller: Any,
        sanitizer: Any,
        debug: bool = False,
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm = llm
        self.tool_caller = tool_caller
        self.sanitizer = sanitizer
        self.debug = debug
        self._schema = schema or {}
        self._resolver = SchemaIntentResolver(self._schema)
        self._system_stage1 = build_system_prompt(schema_text)
        self._logs: List[str] = []

    async def chat(self, user_message: str) -> Dict[str, Any]:
        self._logs = []
        if _is_greeting(user_message):
            return {"reply": _greeting_reply(), "debug": self._logs if self.debug else []}

        tool_call, raw_llm = await self._translate_with_retry(user_message)
        if tool_call is None:
            return {"reply": "Sorry, I could not understand that query. Please try rephrasing.", "debug": self._logs if self.debug else []}

        normalized = _normalize_tool_call(tool_call, schema=self._schema)
        safe = apply_safety_rules(
            normalized["tool"],
            normalized["params"],
            user_message,
            schema=self._schema,
        )
        tool = safe["tool"]
        params = safe["params"]

        try:
            params = self.sanitizer(tool, params)
        except Exception as exc:
            self._log(f"Validation error: {exc}")
            return {"reply": "I need a bit more detail to do that. Please rephrase your request.", "debug": self._logs if self.debug else []}

        self._log(f"Tool: {tool} Params: {json.dumps(params)}")

        try:
            mcp_result = parse_mcp_payload(await self.tool_caller(tool, params))
            self._log(f"MCP returned: {str(mcp_result)[:200]}")
        except Exception as exc:
            self._log(f"MCP error: {exc}")
            return {
                "reply": "Database error. Please try again.",
                "debug": self._logs if self.debug else [],
            }

        human_answer = await self._format_result(user_message, tool, params, mcp_result)
        response: Dict[str, Any] = {"reply": human_answer}
        if self.debug:
            response["tool"] = tool
            response["params"] = params
            response["debug"] = self._logs
        return response

    def _llm_compile_enabled(self) -> bool:
        return str(os.getenv("SHOPPING_LLM_COMPILE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
        }

    async def _translate_with_retry(self, user_message: str) -> Tuple[Optional[Dict[str, Any]], str]:
        resolved = self._resolver.resolve(user_message)
        if resolved:
            self._log(f"Schema intent resolver: {json.dumps(resolved)}")
            return resolved, "(schema-resolver)"

        if not self._llm_compile_enabled():
            self._log("LLM compile disabled (SHOPPING_LLM_COMPILE=0); no resolver match.")
            return None, ""

        raw = await self._llm_call_stage1(user_message)
        self._log(f"LLM raw output (attempt 1): {raw}")
        parsed = extract_json(raw or "")
        valid, err = validate_tool_call(parsed)
        if valid:
            return parsed, raw or ""

        self._log(f"Validation failed: {err}. Retrying...")
        correction_prompt = (
            f"Invalid JSON. Error: {err}\n"
            f"Bad output: {raw}\n"
            f"User: {user_message}\n"
            'Reply with ONE line: {"tool":"query_shopping_trends|summarize_shopping_trends","params":{...}}'
        )
        raw2 = await self._llm_call_raw(correction_prompt)
        self._log(f"LLM raw output (attempt 2): {raw2}")
        parsed2 = extract_json(raw2 or "")
        valid2, err2 = validate_tool_call(parsed2)
        if valid2:
            return parsed2, raw2 or ""

        self._log(f"Both attempts failed. Last error: {err2}")
        return None, raw or ""

    async def _format_result(self, user_message: str, tool: str, params: Dict[str, Any], mcp_result: Any) -> str:
        # Prefer deterministic formatting for common outputs
        formatted = self._format_table(mcp_result)
        if formatted:
            return formatted

        if isinstance(mcp_result, str):
            return mcp_result
        return json.dumps(mcp_result, ensure_ascii=True, indent=2)

    def _format_table(self, mcp_result: Any) -> Optional[str]:
        if isinstance(mcp_result, str):
            if mcp_result.startswith("❌"):
                return "I could not query the database. Check that MongoDB is running and the CSV is imported."
            if mcp_result.startswith("⚠️"):
                return (
                    "No matching records in the shopping trends database. "
                    "Try: import shopping trends (or run the import script), then ask again."
                )
            if "No data found" in mcp_result or "No matching records" in mcp_result:
                return (
                    "No matching records in the shopping trends database. "
                    "Import the dataset first, then retry your question."
                )
        if not mcp_result:
            return "No data found for your query."
        if isinstance(mcp_result, dict) and "count" in mcp_result:
            return f"Total count: {mcp_result['count']:,}"

        if isinstance(mcp_result, list) and mcp_result:
            first = mcp_result[0]
            if not isinstance(first, dict):
                return None

            metric_keys = {"avg_rating", "total_purchases", "revenue", "count"}
            name_key = next((k for k in first.keys() if k not in metric_keys), None)

            has_avg = "avg_rating" in first
            has_count = "total_purchases" in first
            has_rev = "revenue" in first

            if has_avg and has_count and name_key:
                lines = []
                for i, row in enumerate(mcp_result, 1):
                    name = row.get(name_key, "-")
                    rating = row.get("avg_rating", 0)
                    count = int(row.get("total_purchases", 0))
                    lines.append(
                        f"{name}: Average Rating of {rating:.2f} (based on {count:,} reviews)"
                    )
                best = mcp_result[0]
                best_name = best.get(name_key, "-")
                best_val = best.get("avg_rating", 0)
                lines.append(
                    f"\nBest overall: {best_name} (avg rating {best_val:.2f})."
                )
                return "\n".join(lines)

            # Generic table
            cols = list(first.keys())
            header = "| " + " | ".join(_humanize(c) for c in cols) + " |"
            divider = "| " + " | ".join("---" for _ in cols) + " |"
            rows = []
            for row in mcp_result:
                cells = []
                for c in cols:
                    v = row.get(c, "")
                    if isinstance(v, float):
                        cells.append(f"{v:.2f}")
                    else:
                        cells.append(str(v))
                rows.append("| " + " | ".join(cells) + " |")
            return "Here are the results:\n\n" + "\n".join([header, divider] + rows)

        if isinstance(mcp_result, str):
            return mcp_result
        return None

    async def _llm_call_stage1(self, user_message: str) -> Optional[str]:
        try:
            messages = [
                SystemMessage(content=self._system_stage1),
                HumanMessage(content=user_message),
            ]
            response = await self.llm.ainvoke(messages)
            return str(getattr(response, "content", "") or "").strip()
        except Exception as exc:
            self._log(f"LLM call error: {exc}")
            return None

    async def _llm_call_raw(self, prompt: str) -> Optional[str]:
        try:
            messages = [HumanMessage(content=prompt)]
            if "Invalid JSON" in prompt or "Reply with ONE line" in prompt:
                messages = [
                    SystemMessage(content=self._system_stage1),
                    HumanMessage(content=prompt),
                ]
            response = await self.llm.ainvoke(messages)
            return strip_non_json_llm_output(str(getattr(response, "content", "") or ""))
        except Exception as exc:
            self._log(f"LLM call error: {exc}")
            return None

    def _log(self, msg: str) -> None:
        self._logs.append(msg)
        if self.debug:
            print(f"[framework] {msg}")
