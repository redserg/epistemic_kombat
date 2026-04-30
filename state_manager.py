"""Persistence utilities for Epistemic Kombat state."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class GameState(BaseModel):
    current_hp: int
    player_hp: int = 100
    turn_number: int = 1
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    judge_logs: List[Dict[str, Any]] = Field(default_factory=list)
    used_facts: List[str] = Field(default_factory=list)
    session_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, ensure_ascii=False)


def load_state(path: Path, *, fallback: GameState) -> GameState:
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


def archive_game(history_root: Path, state: GameState) -> Path:
    """Copy finished game state into history/<session_id>/game.json."""
    session_dir = history_root / state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / "game.json"
    dest.write_text(state.to_json(), encoding="utf-8")
    return dest


def clear_current(path: Path) -> None:
    """Remove current_game.json so next run starts fresh."""
    if path.exists():
        path.unlink()
