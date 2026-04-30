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
