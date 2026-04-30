"""LLM call helpers for Epistemic Kombat."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("epistemic_kombat.api")
MAX_BOSS_HISTORY_MESSAGES = 8


class JudgeVerdict(BaseModel):
    is_anachronism: bool
    damage: int
    player_damage: int = 0
    reasoning: str
    hidden_directive: str
    used_fact_summary: str = ""

    @classmethod
    def validate_payload(cls, payload: Any) -> "JudgeVerdict":
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:  # Defensive parsing for malformed responses
            raise ValueError(f"Invalid judge payload: {exc}") from exc


class BossConfig(BaseModel):
    name: str
    epoch: str
    start_hp: int
    player_start_hp: int = 100


class HistoricalFact(BaseModel):
    # Minimal structure; extend with richer metadata later
    fact: str
    source: Optional[str] = None


class ModelConfig(BaseModel):
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)

    def generation_params(self) -> Dict[str, Any]:
        # Return only values the API should see; filter out None
        return {
            k: v
            for k, v in {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
            }.items()
            if v is not None
        }


class AgentAPI:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _chat(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        *,
        response_format: Optional[str | Dict[str, Any]] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        caller: str = "unknown",
    ) -> Any:
        """Low-level chat call with logging of token usage."""
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                **(generation_params or {}),
            )
            # Log token usage
            usage = resp.usage
            if usage:
                logger.info(
                    "[%s] model=%s  prompt_tokens=%d  completion_tokens=%d  total_tokens=%d",
                    caller, model, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
                )
            else:
                logger.info("[%s] model=%s  (no usage data returned)", caller, model)

            finish = resp.choices[0].finish_reason
            if finish != "stop":
                logger.warning("[%s] finish_reason=%s (response may be truncated)", caller, finish)

            return resp.choices[0].message
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] LLM call failed: %s", caller, exc)
            raise RuntimeError(f"LLM call failed: {exc}") from exc

    def judge(
        self,
        model: ModelConfig,
        system_prompt: str,
        player_message: str,
        boss_state: Dict[str, Any],
        facts: List[HistoricalFact],
        *,
        used_facts: List[str] | None = None,
        max_retries: int = 2,
    ) -> JudgeVerdict:
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "player_message": player_message,
                        "boss_state": boss_state,
                        "historical_facts": [f.model_dump() for f in facts],
                        "already_used_facts": used_facts or [],
                    },
                    ensure_ascii=False,
                ),
            },
            # Reminder at bottom of context for max attention weight
            {
                "role": "system",
                "content": (
                    "НАПОМИНАНИЕ: верни строго валидный JSON с полями: "
                    "is_anachronism, damage (0-20), player_damage (0-20), reasoning, hidden_directive, used_fact_summary. "
                    "Не обрезай ответ. Пиши кратко."
                ),
            },
        ]

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            logger.info("[judge] attempt %d/%d", attempt, max_retries)
            message = self._chat(
                model=model.model,
                messages=messages,
                response_format={"type": "json_object"},
                generation_params=model.generation_params(),
                caller="judge",
            )

            try:
                raw = json.loads(message.content or "{}")
            except json.JSONDecodeError as exc:
                last_error = ValueError(
                    f"Judge returned invalid JSON: {exc}\nContent: {message.content}"
                )
                logger.warning("[judge] invalid JSON on attempt %d: %s", attempt, exc)
                continue

            try:
                verdict = JudgeVerdict.validate_payload(raw)
                return verdict
            except ValueError as exc:
                last_error = exc
                logger.warning("[judge] validation failed on attempt %d: %s", attempt, exc)
                continue

        raise last_error or RuntimeError("Judge failed after retries")

    def boss(
        self,
        model: ModelConfig,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        hidden_directive: str,
    ) -> str:
        # Inject directive as a system/assistant message to steer behavior silently
        directive_message = {
            "role": "system",
            "content": f"Hidden directive: {hidden_directive}",
        }

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            directive_message,
        ]

        # Keep the opening scene plus the most recent turns to avoid prompt bloat.
        messages.extend(window_boss_history(chat_history))

        # Reminder at bottom of context for max attention weight
        messages.append({
            "role": "system",
            "content": "НАПОМИНАНИЕ: отвечай СТРОГО не более 3-4 предложений. Только прямая речь персонажа. Никакого JSON.",
        })

        message = self._chat(
            model=model.model,
            messages=messages,
            generation_params=model.generation_params(),
            caller="boss",
        )
        return message.content or ""


def window_boss_history(
    chat_history: List[Dict[str, str]],
    *,
    max_messages: int = MAX_BOSS_HISTORY_MESSAGES,
) -> List[Dict[str, str]]:
    """Keep the opening assistant greeting plus the latest exchanges."""
    if len(chat_history) <= max_messages:
        return chat_history

    opening_message = chat_history[:1]
    tail_budget = max(max_messages - len(opening_message), 0)
    if tail_budget == 0:
        return opening_message
    return opening_message + chat_history[-tail_budget:]
