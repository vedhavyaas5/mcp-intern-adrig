from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.query_compiler import build_compiler_prompt, normalize_compiler_output, strip_non_json_llm_output


class LLMMCPBridge:
    def __init__(self, schema_prompt: str, llm: Any, schema: dict | None = None) -> None:
        self.schema_prompt = schema_prompt
        self._schema = schema or {}
        self._system = build_compiler_prompt(schema_prompt)
        self._llm = llm

    async def translate(self, user_message: str) -> Optional[Dict[str, Any]]:
        raw_text = await self._call_llm(user_message)
        if not raw_text:
            return None
        parsed = _parse_json(raw_text)
        if not parsed:
            return None
        return normalize_compiler_output(parsed, schema=self._schema)

    async def _call_llm(self, user_message: str) -> Optional[str]:
        if not self._llm:
            return None

        messages = [
            SystemMessage(content=self._system),
            HumanMessage(content=user_message),
        ]
        response = await self._llm.ainvoke(messages)
        return strip_non_json_llm_output(str(getattr(response, "content", "") or ""))


def normalize_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize JSON output into a {tool_name, tool_args} shape.
    Accepts both 'tool'/'params' and 'tool_name'/'tool_args' formats.
    """
    tool = str(tool_call.get("tool") or tool_call.get("tool_name") or "").strip()
    params = tool_call.get("params")
    if params is None:
        params = tool_call.get("tool_args")
    if not isinstance(params, dict):
        params = {}

    normalized = normalize_compiler_output({"tool": tool, "params": params})
    return {
        "tool_name": normalized["tool"],
        "tool_args": normalized["params"],
    }


def coerce_query_json(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, str):
        return value.strip() or "{}"
    return "{}"


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from the LLM output.
    Handles:
      - bare JSON
      - ```json ... ``` fences
      - JSON embedded in prose
    """
    text = strip_non_json_llm_output(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    return None
