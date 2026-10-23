"""Structured outputs of the two LLM calls. Every field has a non-null default so the
schemas work with strict JSON-schema modes (no nulls; "" means "not given")."""
from __future__ import annotations

import copy
import json
import re
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError

Intent = Literal[
    "policy_question",
    "balance_query",
    "requests_query",
    "team_leave_query",
    "apply_leave",
    "cancel_leave",
    "future_balance",
    "contact_info",
    "smalltalk",
    "out_of_scope",
    "manipulation",
]


class RouterOutput(BaseModel):
    intent: Intent = "policy_question"
    bulk_request: bool = False
    target_kind: Literal["self", "employee_id", "name", "none"] = "self"
    target_value: str = ""
    leave_type: Literal["annual", "sick", ""] = ""
    status_filter: Literal["pending", "approved", "cancelled", ""] = ""
    start_date: str = ""
    end_date: str = ""
    date_phrase: str = ""
    dates_ambiguous: bool = False
    half_day: bool = False
    request_id: str = ""
    cancel_hint: str = ""
    past_reference: str = ""
    target_date: str = ""
    needs_documents: bool = False
    standalone_question: str = ""
    search_queries: list[str] = Field(default_factory=list)
    refers_to_pending: bool = False


class ComposerOutput(BaseModel):
    answer: str = ""
    used_sources: list[str] = Field(default_factory=list)
    coverage: Literal["full", "partial", "none"] = "none"


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema -> strict form: every property required, no extra keys, no defaults/titles."""
    schema = copy.deepcopy(model.model_json_schema())

    def clean(node):
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False
            for value in node.values():
                clean(value)
        elif isinstance(node, list):
            for value in node:
                clean(value)

    clean(schema)
    return schema


class LLMOutputError(Exception):
    pass


def parse_json_lenient(text: str, model: type[BaseModel]) -> BaseModel:
    """Accept raw JSON, fenced JSON, or JSON embedded in prose; validate with defaults."""
    if text is None:
        raise LLMOutputError("empty response")
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S)
    if fence:
        candidate = fence.group(1).strip()
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputError("no JSON object in response")
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise LLMOutputError("JSON is not an object")
    # Tolerate nulls and out-of-vocabulary enum values from providers that ignore the
    # schema: drop them so the field defaults apply.
    cleaned = {}
    for key, value in data.items():
        field = model.model_fields.get(key)
        if value is None or field is None:
            continue
        if get_origin(field.annotation) is Literal and value not in get_args(field.annotation):
            continue
        cleaned[key] = value
    data = cleaned
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError(f"schema validation failed: {exc.errors()[:2]}") from None
