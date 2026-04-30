"""LLM call helpers for Epistemic Kombat."""
from __future__ import annotations

from difflib import SequenceMatcher
import json
import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("epistemic_kombat.api")
MAX_BOSS_HISTORY_MESSAGES = 8
DEFAULT_BOSS_RESPONSE_TOKENS = 160
DEFAULT_JUDGE_RESPONSE_TOKENS = 220


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

    def normalized(self, *, used_facts: List[str] | None = None, locale: str = "ru") -> "JudgeVerdict":
        """Smooth out contradictory scoring before gameplay logic consumes it."""
        updates: Dict[str, Any] = {}
        if not self.is_anachronism and self.damage >= 12 and self.player_damage > 0:
            updates["player_damage"] = 0
        if used_fact_summary_is_repeat(self.used_fact_summary, used_facts or []):
            updates["damage"] = min(self.damage, 4)
            updates["player_damage"] = max(self.player_damage, 3)
            updates["used_fact_summary"] = ""
            updates["hidden_directive"] = default_hidden_directive(
                locale=locale,
                damage=updates["damage"],
                player_damage=updates["player_damage"],
                is_anachronism=self.is_anachronism,
            )
        return self.model_copy(update=updates) if updates else self


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


class ChatResult(BaseModel):
    content: str
    reasoning: str = ""
    finish_reason: str = "stop"


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

            message = resp.choices[0].message
            return ChatResult(
                content=message.content or "",
                reasoning=getattr(message, "reasoning", None) or "",
                finish_reason=finish,
            )
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
        llm_mode: str = "cloud",
        locale: str = "ru",
        used_facts: List[str] | None = None,
        max_retries: int = 2,
        response_token_limit: int = DEFAULT_JUDGE_RESPONSE_TOKENS,
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
                "content": judge_json_reminder(locale),
            },
        ]

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            logger.info("[judge] attempt %d/%d", attempt, max_retries)
            attempt_messages = list(messages)
            if attempt > 1:
                attempt_messages.append(
                    {
                        "role": "system",
                        "content": judge_retry_reminder(locale, llm_mode=llm_mode),
                    }
                )
            result = self._chat(
                model=model.model,
                messages=attempt_messages,
                response_format=judge_response_format_for_attempt(llm_mode=llm_mode, attempt=attempt),
                generation_params=judge_generation_params_for_attempt(
                    model.generation_params(),
                    response_token_limit,
                    attempt,
                ),
                caller="judge",
            )

            try:
                raw = json.loads(extract_json_object(result.content or "{}"))
            except json.JSONDecodeError as exc:
                last_error = ValueError(
                    f"Judge returned invalid JSON: {exc}\nContent: {result.content}"
                )
                logger.warning("[judge] invalid JSON on attempt %d: %s", attempt, exc)
                continue

            try:
                verdict = JudgeVerdict.validate_payload(raw).normalized(
                    used_facts=used_facts,
                    locale=locale,
                )
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
        *,
        locale: str = "ru",
        max_retries: int = 2,
        response_token_limit: int = DEFAULT_BOSS_RESPONSE_TOKENS,
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
            "content": boss_reply_reminder(locale),
        })

        for attempt in range(1, max_retries + 1):
            attempt_messages = list(messages)
            if attempt > 1:
                attempt_messages.append(
                    {
                        "role": "system",
                        "content": boss_retry_reminder(locale),
                    }
                )

            result = self._chat(
                model=model.model,
                messages=attempt_messages,
                generation_params=boss_generation_params_for_attempt(
                    model.generation_params(),
                    boss_response_token_limit_for_attempt(response_token_limit, attempt),
                    attempt,
                ),
                caller="boss",
            )
            cleaned_reply = clean_boss_reply(result.content)
            if boss_reply_is_usable(cleaned_reply, result.finish_reason):
                return cleaned_reply

        return boss_fallback_reply(locale)


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


def limit_max_tokens(generation_params: Dict[str, Any], token_limit: int) -> Dict[str, Any]:
    """Clamp response size for tightly structured model calls."""
    params = dict(generation_params)
    params["max_tokens"] = min(params.get("max_tokens", token_limit), token_limit)
    return params


def extract_json_object(content: str) -> str:
    """Extract the first balanced JSON object from a model response."""
    start = content.find("{")
    if start == -1:
        return content

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(content)):
        char = content[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]

    return content[start:]


def normalize_fact_summary(summary: str) -> str:
    """Normalize short fact summaries before fuzzy repeat matching."""
    tokens = re.findall(r"\w+", summary.lower())
    return " ".join(tokens)


def used_fact_summary_is_repeat(summary: str, used_facts: List[str]) -> bool:
    """Catch obvious paraphrase repeats that the judge failed to downscore."""
    normalized_summary = normalize_fact_summary(summary)
    if not normalized_summary:
        return False

    for previous in used_facts:
        normalized_previous = normalize_fact_summary(previous)
        if not normalized_previous:
            continue
        similarity = SequenceMatcher(None, normalized_summary, normalized_previous).ratio()
        if similarity >= 0.55:
            return True
    return False


def clean_boss_reply(content: str) -> str:
    """Normalize boss output and reject empty replies."""
    cleaned = " ".join(content.split()).strip()
    return cleaned


def boss_reply_is_usable(reply: str, finish_reason: str) -> bool:
    """Accept only complete-looking boss replies."""
    if not reply:
        return False
    if finish_reason != "stop":
        return False
    return reply[-1] in ".!?"


def boss_response_token_limit_for_attempt(response_token_limit: int, attempt: int) -> int:
    """Use a stricter cap on retries to force concise completed boss replies."""
    if attempt <= 1:
        return response_token_limit
    return min(response_token_limit, 220)


def boss_generation_params_for_attempt(
    generation_params: Dict[str, Any],
    token_limit: int,
    attempt: int,
) -> Dict[str, Any]:
    """Retries should be shorter and calmer than the initial boss answer."""
    params = limit_max_tokens(generation_params, token_limit)
    if attempt > 1:
        params["temperature"] = min(params.get("temperature", 0.7), 0.4)
    return params


def judge_generation_params_for_attempt(
    generation_params: Dict[str, Any],
    token_limit: int,
    attempt: int,
) -> Dict[str, Any]:
    """Retries for judge should be compact and deterministic."""
    params = limit_max_tokens(generation_params, token_limit)
    if attempt > 1:
        params["temperature"] = min(params.get("temperature", 0.2), 0.1)
    return params


def judge_response_format_for_attempt(*, llm_mode: str, attempt: int) -> Optional[Dict[str, str]]:
    """Local retries fall back to plain JSON generation when structured mode is flaky."""
    if llm_mode == "local" and attempt > 1:
        return None
    return {"type": "json_object"}


def stabilize_hidden_directive(
    directive: str,
    *,
    locale: str,
    remaining_boss_hp: int,
    damage: int,
    player_damage: int,
    is_anachronism: bool,
) -> str:
    """Prevent premature boss capitulation before a stage is actually won."""
    cleaned = " ".join(directive.split()).strip()
    if remaining_boss_hp <= 0:
        return cleaned

    if is_hidden_directive_safe(cleaned):
        return cleaned

    return default_hidden_directive(
        locale=locale,
        damage=damage,
        player_damage=player_damage,
        is_anachronism=is_anachronism,
    )


def is_hidden_directive_safe(directive: str) -> bool:
    """Keep only directives that clearly preserve opposition or tone."""
    if not directive:
        return False
    lowered = directive.lower()
    unsafe_markers = (
        "accept",
        "agree",
        "confirm",
        "support the argument",
        "argue for",
        "curvature",
        "spherical earth",
        "external light",
        "прими",
        "соглас",
        "подтверди",
        "поддержи аргумент",
        "аргументом о кривизне",
        "шарообраз",
        "внешн",
    )
    if any(marker in lowered for marker in unsafe_markers):
        return False

    safe_markers = (
        "defend",
        "resist",
        "justify",
        "dismiss",
        "mock",
        "pressure",
        "защищ",
        "сопротив",
        "оправд",
        "высмей",
        "отверг",
        "оспор",
    )
    return any(marker in lowered for marker in safe_markers)


def default_hidden_directive(
    *,
    locale: str,
    damage: int,
    player_damage: int,
    is_anachronism: bool,
) -> str:
    """Provide a robust fallback directive when judge wording is unsafe."""
    if locale == "en":
        if is_anachronism:
            return "Mock the anachronism and reject it sharply."
        if damage >= 12:
            return "Admit the observation is powerful, but resist and defend your worldview."
        if player_damage >= 7 and damage <= 3:
            return "Dismiss the argument sharply and press your advantage."
        if damage >= 5:
            return "Acknowledge the pressure, but explain the observation away and keep resisting."
        return "Hold your position and demand clearer evidence."

    if is_anachronism:
        return "Высмей анахронизм и резко отвергни довод."
    if damage >= 12:
        return "Признай силу наблюдения, но продолжай защищать свою картину мира."
    if player_damage >= 7 and damage <= 3:
        return "Резко отвергни довод и надави на слабость аргумента."
    if damage >= 5:
        return "Покажи давление аргумента, но объясни наблюдение по-своему и продолжай сопротивляться."
    return "Стой на своём и требуй более ясных доказательств."


def judge_json_reminder(locale: str) -> str:
    if locale == "en":
        return (
            "REMINDER: return strictly valid JSON with fields "
            "is_anachronism, damage (0-20), player_damage (0-20), reasoning, hidden_directive, used_fact_summary. "
            "Keep it short and do not truncate the answer."
        )
    return (
        "НАПОМИНАНИЕ: верни строго валидный JSON с полями "
        "is_anachronism, damage (0-20), player_damage (0-20), reasoning, hidden_directive, used_fact_summary. "
        "Не обрезай ответ. Пиши кратко."
    )


def judge_retry_reminder(locale: str, *, llm_mode: str = "cloud") -> str:
    if locale == "en":
        if llm_mode == "local":
            return "THE PREVIOUS ANSWER FAILED. Return ONLY a short JSON object in normal message content, with no surrounding text."
        return "THE PREVIOUS ANSWER WAS INVALID OR TRUNCATED. Return only a short JSON object with no surrounding text."
    if llm_mode == "local":
        return "ПРЕДЫДУЩИЙ ОТВЕТ НЕ ПОДОШЁЛ. Верни ТОЛЬКО короткий JSON-объект обычным текстом ответа, без пояснений вокруг."
    return "ПРЕДЫДУЩИЙ ОТВЕТ БЫЛ НЕВАЛИДЕН ИЛИ ОБРЕЗАН. Верни только короткий JSON-объект без пояснений вокруг него."


def boss_reply_reminder(locale: str) -> str:
    if locale == "en":
        return "REMINDER: reply in strictly 2-4 short sentences. Direct speech only. No JSON."
    return "НАПОМИНАНИЕ: отвечай СТРОГО не более 3-4 предложений. Только прямая речь персонажа. Никакого JSON."


def boss_retry_reminder(locale: str) -> str:
    if locale == "en":
        return (
            "THE PREVIOUS ANSWER WAS EMPTY OR CUT OFF. "
            "Return at most 2 short sentences in English, under 45 words total, direct speech only."
        )
    return (
        "ПРЕДЫДУЩИЙ ОТВЕТ ПУСТ ИЛИ ОБОРВАН. "
        "Верни не более 2 коротких предложений на русском, суммарно до 45 слов, только прямую речь персонажа."
    )


def boss_fallback_reply(locale: str) -> str:
    if locale == "en":
        return "I fall silent for a moment to gather my thoughts. Repeat your argument once more, and I will answer more clearly."
    return "Я на миг умолк, собирая мысли. Повтори свой довод еще раз, и я отвечу яснее."
