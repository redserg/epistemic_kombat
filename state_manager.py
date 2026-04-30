"""Persistence utilities for Epistemic Kombat state."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping

from pydantic import BaseModel, Field


def new_session_id() -> str:
    """Generate a readable session id with enough precision to avoid collisions."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


class GameState(BaseModel):
    locale: str = "ru"
    campaign_id: str = "earth_shape"
    stage_index: int = 0
    current_hp: int
    player_hp: int = 100
    turn_number: int = 1
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    judge_logs: List[Dict[str, Any]] = Field(default_factory=list)
    used_facts: List[str] = Field(default_factory=list)
    completed_stages: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: str = Field(default_factory=new_session_id)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, ensure_ascii=False)


def game_outcome(state: GameState) -> Literal["victory", "defeat"] | None:
    """Return terminal outcome for finished games, otherwise None."""
    if state.current_hp <= 0:
        return "victory"
    if state.player_hp <= 0:
        return "defeat"
    return None


def advance_turn(state: GameState) -> None:
    """Persist the next turn number after a completed exchange."""
    state.turn_number += 1


def reset_stage_progress(state: GameState, *, boss_hp: int, player_hp: int) -> None:
    """Prepare state for a fresh stage while keeping session-wide metadata."""
    state.current_hp = boss_hp
    state.player_hp = player_hp
    state.turn_number = 1
    state.chat_history.clear()
    state.judge_logs.clear()
    state.used_facts.clear()


def load_state(path: Path, *, fallback: GameState | None = None) -> GameState | None:
    if not path.exists():
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return GameState.model_validate(data)
    except Exception:  # noqa: BLE001
        # Corrupt or incompatible state -> restart with fallback
        return fallback


def save_state(path: Path, state: GameState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json(), encoding="utf-8")


def render_transcript_markdown(
    state: GameState,
    *,
    status: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Render a readable transcript for humans inspecting playtest history."""
    metadata = metadata or {}
    lines = [
        f"# Session {state.session_id}",
        "",
    ]

    if status:
        lines.append(f"- Status: {status}")
    if metadata.get("llm_mode"):
        lines.append(f"- LLM mode: {metadata['llm_mode']}")
    if metadata.get("campaign_title"):
        lines.append(f"- Campaign title: {metadata['campaign_title']}")
    lines.extend(
        [
        f"- Locale: {state.locale}",
        f"- Campaign: {state.campaign_id}",
        f"- Stage title: {metadata.get('stage_title', 'unknown')}",
        f"- Stage index: {state.stage_index}",
        f"- Boss: {metadata.get('boss_name', 'unknown')}",
        f"- Boss HP: {state.current_hp}",
        f"- Player HP: {state.player_hp}",
        f"- Turn number: {state.turn_number}",
        "",
        ]
    )

    if state.completed_stages:
        lines.extend(["## Completed Stages", ""])
        for index, completed in enumerate(state.completed_stages, start=1):
            lines.append(f"### Stage {index}: {completed.get('stage_title', completed.get('stage_id', 'unknown'))}")
            lines.append("")
            lines.append(f"- Outcome: {completed.get('outcome', 'unknown')}")
            lines.append(f"- Boss HP: {completed.get('boss_hp', '?')}")
            lines.append(f"- Player HP: {completed.get('player_hp', '?')}")
            lines.append(f"- Turn number: {completed.get('turn_number', '?')}")
            lines.append("")

            lines.append("#### Dialogue")
            lines.append("")
            for message in completed.get("chat_history", []):
                role = message.get("role", "unknown")
                content = message.get("content", "").strip()
                speaker = "Boss" if role == "assistant" else "Player" if role == "user" else role.capitalize()
                lines.append(f"##### {speaker}")
                lines.append("")
                lines.append(content or "[empty]")
                lines.append("")

            judge_logs = completed.get("judge_logs", [])
            if judge_logs:
                lines.append("#### Judge Logs")
                lines.append("")
                for entry in judge_logs:
                    lines.append(f"- Turn {entry.get('turn', '?')}: {entry.get('verdict', {}).get('reasoning', '')}")
                lines.append("")

    if state.judge_logs:
        lines.extend(["## Turn Summary", ""])
        for entry in state.judge_logs:
            verdict = entry.get("verdict", {})
            lines.append(
                f"- Turn {entry.get('turn', '?')}: boss -{verdict.get('damage', 0)}, "
                f"player -{verdict.get('player_damage', 0)} | {verdict.get('reasoning', '')}"
            )
        lines.append("")

    lines.extend(["## Dialogue", ""])

    for message in state.chat_history:
        role = message.get("role", "unknown")
        content = message.get("content", "").strip()
        if role == "assistant":
            speaker = "Boss"
        elif role == "user":
            speaker = "Player"
        else:
            speaker = role.capitalize()
        lines.append(f"### {speaker}")
        lines.append("")
        lines.append(content or "[empty]")
        lines.append("")

    if state.judge_logs:
        lines.extend(["## Judge Logs", ""])
        for entry in state.judge_logs:
            lines.append(f"### Turn {entry.get('turn', '?')}")
            lines.append("")
            verdict = entry.get("verdict", {})
            for key, value in verdict.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_report_markdown(
    state: GameState,
    *,
    status: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    log_text: str = "",
) -> str:
    """Render a short Russian report that explains why a run is interesting."""
    metadata = metadata or {}
    judge_logs = state.judge_logs
    fallback_count = sum(
        1
        for entry in judge_logs
        if "fallback: judge error" in entry.get("verdict", {}).get("reasoning", "").lower()
    )
    anachronism_count = sum(
        1 for entry in judge_logs if entry.get("verdict", {}).get("is_anachronism")
    )
    repeat_penalty_count = sum(
        1
        for entry in judge_logs
        if "repetition" in entry.get("verdict", {}).get("reasoning", "").lower()
        or "повтор" in entry.get("verdict", {}).get("reasoning", "").lower()
    )
    collision_markers = log_text.count("Session started:")

    highlights: List[str] = []
    problems: List[str] = []

    if state.completed_stages:
        cleared_titles = ", ".join(
            completed.get("stage_title", completed.get("stage_id", "unknown"))
            for completed in state.completed_stages
        )
        highlights.append(f"Ран успел завершить уровни: {cleared_titles}.")
    if state.turn_number >= 7:
        highlights.append(
            f"Это длинный прогон: сыграно как минимум {state.turn_number - 1} полных ходов."
        )
    if status == "paused" and 0 < state.current_hp <= 20:
        highlights.append(
            f"Игра поставлена на паузу почти у финиша: у босса осталось всего {state.current_hp} HP."
        )
    if status == "paused" and 0 < state.player_hp <= 20:
        highlights.append(
            f"Игра прервана на грани поражения игрока: осталось только {state.player_hp} HP."
        )
    if state.stage_index > 0:
        highlights.append(
            f"Сессия уже добралась до этапа {state.stage_index + 1}, так что видно переход между уровнями."
        )

    if fallback_count:
        problems.append(
            f"Зафиксировано {fallback_count} judge fallback-эпизод(а), где локальная модель не вернула пригодный JSON."
        )
    if anachronism_count:
        problems.append(
            f"В споре случилось {anachronism_count} отметок анахронизма; полезно проверить, были ли они справедливы."
        )
    if repeat_penalty_count:
        problems.append(
            f"Сработало {repeat_penalty_count} штрафов за повтор аргумента; это хороший тест анти-спам логики."
        )
    if collision_markers > 1:
        problems.append(
            "В `game.log` видно несколько стартов одной и той же сессии: это признак коллизии `session_id` при параллельном запуске."
        )

    if not highlights:
        highlights.append("Это спокойный базовый ран без редких событий, удобный как контрольный пример.")
    if not problems:
        problems.append("Явных аварийных симптомов в этом ране не видно; он полезен как эталон нормальной игры.")

    lines = [
        f"# Отчёт по рану {state.session_id}",
        "",
        "## Сводка",
        "",
        f"- Статус: {status or 'unknown'}",
        f"- Кампания: {metadata.get('campaign_title', state.campaign_id)}",
        f"- Уровень: {metadata.get('stage_title', 'unknown')}",
        f"- Локаль: {state.locale}",
        f"- LLM-режим: {metadata.get('llm_mode', 'unknown')}",
        f"- Ход: {state.turn_number}",
        f"- HP босса: {state.current_hp}",
        f"- HP игрока: {state.player_hp}",
        "",
        "## Чем интересен этот ран",
        "",
    ]
    lines.extend(f"- {item}" for item in highlights)
    lines.extend(["", "## Что стоит заметить", ""])
    lines.extend(f"- {item}" for item in problems)
    lines.extend(["", "## Зачем хранить этот пример", ""])

    if collision_markers > 1:
        lines.append("- Это хороший артефакт для отладки параллельных запусков и сохранения истории.")
    elif fallback_count or repeat_penalty_count or anachronism_count:
        lines.append("- Этот ран показывает не только диалог, но и поведение защитных механизмов судьи и игрового цикла.")
    else:
        lines.append("- Этот ран полезен как человечески читаемый пример того, как выглядит нормальная партия в текущей версии.")

    return "\n".join(lines).strip() + "\n"


def save_history_snapshot(
    history_root: Path,
    state: GameState,
    *,
    status: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a full session snapshot and transcript under history/<session_id>/."""
    session_dir = history_root / state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})

    game_payload = {
        "status": status,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata,
        "state": state.model_dump(),
    }
    (session_dir / "game.json").write_text(
        json.dumps(game_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (session_dir / "transcript.md").write_text(
        render_transcript_markdown(state, status=status, metadata=metadata),
        encoding="utf-8",
    )
    log_text = ""
    log_path = session_dir / "game.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8")
    (session_dir / "report.md").write_text(
        render_report_markdown(state, status=status, metadata=metadata, log_text=log_text),
        encoding="utf-8",
    )
    return session_dir


def archive_game(
    history_root: Path,
    state: GameState,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Copy finished game state into history/<session_id>/game.json."""
    return save_history_snapshot(history_root, state, status="finished", metadata=metadata)


def clear_current(path: Path) -> None:
    """Remove current_game.json so next run starts fresh."""
    if path.exists():
        path.unlink()
