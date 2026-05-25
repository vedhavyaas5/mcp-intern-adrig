from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from pymongo import MongoClient


class SchemaInspector:
    """
    Samples a MongoDB collection and infers a compact schema:
      - column names
      - type (number | string | boolean)
      - for numbers: min, max, p20, p80
      - for strings: up to 20 unique values (if cardinality is low)
    """

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        collection_name: str | None = None,
        sample_size: int = 500,
    ) -> None:
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name or os.getenv("MONGODB_DB", "intern_adrig")
        self.col_name = collection_name or os.getenv("MONGODB_COLLECTION", "shopping_trends")
        self.sample_size = sample_size
        self._schema: Optional[Dict[str, Any]] = None

    def build(self) -> Dict[str, Any]:
        """
        Synchronous build (call once at startup).
        Returns the schema dict and caches it internally.
        """
        client = MongoClient(self.uri)
        col = client[self.db_name][self.col_name]

        docs = list(
            col.aggregate(
                [
                    {"$sample": {"size": self.sample_size}},
                    {"$project": {"_id": 0}},
                ]
            )
        )

        if not docs:
            self._schema = {}
            client.close()
            return {}

        field_values: Dict[str, List[Any]] = defaultdict(list)
        for doc in docs:
            for key, value in doc.items():
                if value is not None:
                    field_values[key].append(value)

        schema: Dict[str, Any] = {}
        for field, values in field_values.items():
            schema[field] = self._infer_field(values)

        self._schema = schema
        client.close()
        return schema

    def _infer_field(self, values: List[Any]) -> Dict[str, Any]:
        numeric_vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        bool_vals = [v for v in values if isinstance(v, bool)]
        string_vals = [str(v) for v in values if isinstance(v, str)]

        if len(bool_vals) / max(len(values), 1) > 0.7:
            return {"type": "boolean"}

        if len(numeric_vals) / max(len(values), 1) > 0.7:
            sorted_v = sorted(numeric_vals)
            n = len(sorted_v)
            return {
                "type": "number",
                "min": round(sorted_v[0], 2),
                "max": round(sorted_v[-1], 2),
                "p20": round(sorted_v[int(n * 0.20)], 2),
                "p80": round(sorted_v[int(n * 0.80)], 2),
            }

        unique = list(dict.fromkeys(string_vals))
        if 1 < len(unique) <= 20:
            return {"type": "string", "values": unique[:20]}

        return {"type": "string"}

    def to_prompt(self) -> str:
        """
        Convert the schema to a compact prompt string.

        Example:
          - review_rating: number [1.0-5.0] top>=4.2 bottom<=2.1
          - category: string values: Clothing, Footwear
        """
        if not self._schema:
            return "Schema not yet loaded."

        lines = [f"Collection: {self.col_name}", "Columns:"]
        for field, meta in self._schema.items():
            if meta.get("type") == "number":
                line = (
                    f"  - {field}: number "
                    f"[{meta['min']}-{meta['max']}] "
                    f"top>={meta['p80']} bottom<={meta['p20']}"
                )
            elif meta.get("type") == "boolean":
                line = f"  - {field}: boolean"
            elif meta.get("values"):
                vals = ", ".join(meta["values"][:10])
                line = f"  - {field}: string values: {vals}"
            else:
                line = f"  - {field}: string"
            lines.append(line)

        return "\n".join(lines)

    def schema(self) -> Optional[Dict[str, Any]]:
        return self._schema
