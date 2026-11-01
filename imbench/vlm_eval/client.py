"""OpenAI Responses API client for the VLM eval (Structured Outputs, retries, cost, mock).

GPT-6 Astra does not accept temperature/top_p, so they are never sent.
Reasoning effort: the highest value in the installed SDK's ReasoningEffort literal
is "max". Not every model supports every value, so the client requests "max" and
steps down ("xhigh", "high") only if the API rejects the effort with a 400. The
effort actually used is recorded in every result.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

MODEL = "gpt-6-astra"
PRICE_PER_M_INPUT = 10.0    # USD per 1M input tokens
PRICE_PER_M_OUTPUT = 50.0   # USD per 1M output tokens (reasoning tokens included)
EFFORT_CHAIN = ("max", "xhigh", "high")          # Stage 1/2: highest supported effort
LOWEST_EFFORT_CHAIN = ("low", "medium")  # Stage 3: lowest effort Astra accepts (API: low|medium|high|xhigh|max)
MAX_RETRIES = 3
KEY_FILE = Path.home() / "openai_astra_key.txt"


class QuotaExhausted(RuntimeError):
    """429 insufficient_quota / no credits: not transient, never retried."""


def _quota_exhausted(e) -> bool:
    msg = str(e).lower()
    return "insufficient_quota" in msg or "no credits" in msg or "exceeded your current quota" in msg


class BudgetExceeded(RuntimeError):
    pass


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1e6 * PRICE_PER_M_INPUT + output_tokens / 1e6 * PRICE_PER_M_OUTPUT


def image_part(path: str | Path, detail: str = "high") -> dict:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": detail}


def text_part(text: str) -> dict:
    return {"type": "input_text", "text": text}


@dataclass
class CallResult:
    parsed: BaseModel | None
    raw: dict
    usage: dict
    model_snapshot: str
    effort: str | None
    cost_usd: float
    error: str | None = None


@dataclass
class Ledger:
    """Tracks spend for one runner invocation and enforces the budget."""
    budget_usd: float = float("inf")
    spent_usd: float = 0.0
    calls: int = 0

    def check(self):
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(f"spent ${self.spent_usd:.2f} >= budget ${self.budget_usd:.2f}")

    def add(self, c: float):
        self.spent_usd += c
        self.calls += 1


def _usage_dict(u) -> dict:
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0}
    d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
    return d


class VLMClient:
    def __init__(self, model: str = MODEL, mock: bool = False, ledger: Ledger | None = None,
                 mock_fn: Callable[[str, type[BaseModel], int], BaseModel] | None = None,
                 effort_chain: tuple[str, ...] = EFFORT_CHAIN):
        self.model = model
        self.effort_chain = tuple(effort_chain)
        self.mock = mock
        self.ledger = ledger or Ledger()
        self.mock_fn = mock_fn
        self._effort_idx = 0
        self._summary = True       # request a reasoning summary; dropped if the API rejects it
        self._mock_calls = 0
        self.mock_cost_usd = 0.0   # per-call cost charged in mock mode (for budget tests)
        if not mock:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key and KEY_FILE.exists():
                key = KEY_FILE.read_text().strip()
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set and ~/openai_astra_key.txt missing")
            self._client = OpenAI(api_key=key, max_retries=0)

    @property
    def effort(self) -> str:
        return self.effort_chain[self._effort_idx]

    def _effort_rejected(self, msg: str) -> bool:
        # e.g. "Unsupported value: 'minimal' is not supported with the 'gpt-6-astra' model."
        return "effort" in msg or (f"'{self.effort}'" in msg and "not supported" in msg)

    def reasoning_param(self) -> dict:
        r = {"effort": self.effort}
        if self._summary:
            r["summary"] = "auto"
        return r

    def call(self, system: str, content: list[dict], schema: type[BaseModel], tag: str = "") -> CallResult:
        self.ledger.check()
        if self.mock:
            return self._mock_call(schema, tag)

        import openai
        last_err = None
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                resp = self._client.responses.parse(
                    model=self.model,
                    instructions=system,
                    input=[{"role": "user", "content": content}],
                    reasoning=self.reasoning_param(),
                    text_format=schema,
                    store=False,
                )
            except openai.BadRequestError as e:
                msg = str(e).lower()
                if "summary" in msg and self._summary:
                    self._summary = False
                    continue
                if self._effort_rejected(msg) and self._effort_idx + 1 < len(self.effort_chain):
                    self._effort_idx += 1          # rejected before generation, not billed
                    continue
                raise
            except openai.RateLimitError as e:
                if _quota_exhausted(e):
                    raise QuotaExhausted(f"OpenAI account out of credits/quota: {e}") from e
                last_err = e
                attempt += 1
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            except (openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as e:
                last_err = e
                attempt += 1
                time.sleep(min(60, 5 * 2 ** attempt))
                continue

            usage = _usage_dict(resp.usage)
            c = cost_usd(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            self.ledger.add(c)
            raw = json.loads(resp.model_dump_json())
            parsed = getattr(resp, "output_parsed", None)
            err = None
            if parsed is None:
                err = f"no parsed output (status={resp.status}, incomplete={resp.incomplete_details})"
            return CallResult(parsed, raw, usage, resp.model, self.effort, c, err)
        raise RuntimeError(f"API call failed after {MAX_RETRIES} retries: {last_err}")

    def call_tools(self, system: str, content: list[dict], tools: list[dict]) -> dict:
        """One Responses API call that must return a function call (tool_choice=required).

        Returns {raw, usage, cost_usd, model_snapshot, effort, tool_name, arguments_json, reasoning_summary}.
        Raises after MAX_RETRIES transient failures (caller falls back to HOLD).
        """
        self.ledger.check()
        import openai
        last_err, attempt = None, 0
        while attempt < MAX_RETRIES:
            try:
                resp = self._client.responses.create(
                    model=self.model, instructions=system,
                    input=[{"role": "user", "content": content}],
                    tools=tools, tool_choice="required", parallel_tool_calls=False,
                    reasoning=self.reasoning_param(), store=False,
                )
            except openai.BadRequestError as e:
                msg = str(e).lower()
                if "summary" in msg and self._summary:
                    self._summary = False
                    continue
                if self._effort_rejected(msg) and self._effort_idx + 1 < len(self.effort_chain):
                    self._effort_idx += 1
                    continue
                raise
            except openai.RateLimitError as e:
                if _quota_exhausted(e):
                    raise QuotaExhausted(f"OpenAI account out of credits/quota: {e}") from e
                last_err = e
                attempt += 1
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            except (openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as e:
                last_err = e
                attempt += 1
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            usage = _usage_dict(resp.usage)
            c = cost_usd(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            self.ledger.add(c)
            raw = json.loads(resp.model_dump_json())
            fc = next((o for o in raw.get("output", []) if o.get("type") == "function_call"), None)
            summ = [s.get("text", "") for o in raw.get("output", []) if o.get("type") == "reasoning"
                    for s in (o.get("summary") or [])]
            return {"raw": raw, "usage": usage, "cost_usd": c, "model_snapshot": resp.model,
                    "effort": self.effort, "tool_name": fc.get("name") if fc else None,
                    "arguments_json": fc.get("arguments") if fc else None, "reasoning_summary": summ}
        raise RuntimeError(f"API call failed after {MAX_RETRIES} retries: {last_err}")

    def _mock_call(self, schema, tag) -> CallResult:
        from imbench.vlm_eval import mock
        self._mock_calls += 1
        parsed = (self.mock_fn or mock.canned)(tag, schema, self._mock_calls)
        usage = {"input_tokens": 12000, "output_tokens": 3000, "mock": True}
        self.ledger.add(self.mock_cost_usd)
        return CallResult(parsed, {"mock": True, "output_parsed": parsed.model_dump()}, usage,
                          f"{self.model}-MOCK", self.effort, self.mock_cost_usd)
