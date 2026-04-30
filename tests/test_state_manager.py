import json
import tempfile
import unittest
from pathlib import Path

from state_manager import (
    GameState,
    advance_turn,
    game_outcome,
    new_session_id,
    render_transcript_markdown,
    reset_stage_progress,
    save_history_snapshot,
)


class GameOutcomeTests(unittest.TestCase):
    def test_new_session_id_is_precise_and_unique(self) -> None:
        first = new_session_id()
        second = new_session_id()

        self.assertRegex(first, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}$")
        self.assertNotEqual(first, second)

    def test_returns_victory_when_boss_hp_is_zero(self) -> None:
        state = GameState(current_hp=0, player_hp=42)
        self.assertEqual(game_outcome(state), "victory")

    def test_returns_defeat_when_player_hp_is_zero(self) -> None:
        state = GameState(current_hp=13, player_hp=0)
        self.assertEqual(game_outcome(state), "defeat")

    def test_returns_none_for_in_progress_game(self) -> None:
        state = GameState(current_hp=13, player_hp=42)
        self.assertIsNone(game_outcome(state))

    def test_advance_turn_moves_state_to_next_prompt_number(self) -> None:
        state = GameState(current_hp=13, player_hp=42, turn_number=3)
        advance_turn(state)
        self.assertEqual(state.turn_number, 4)

    def test_reset_stage_progress_clears_stage_specific_fields(self) -> None:
        state = GameState(
            current_hp=13,
            player_hp=42,
            turn_number=5,
            chat_history=[{"role": "user", "content": "arg"}],
            judge_logs=[{"turn": 1, "verdict": {"damage": 3}}],
            used_facts=["round shadow"],
        )

        reset_stage_progress(state, boss_hp=100, player_hp=90)

        self.assertEqual(state.current_hp, 100)
        self.assertEqual(state.player_hp, 90)
        self.assertEqual(state.turn_number, 1)
        self.assertEqual(state.chat_history, [])
        self.assertEqual(state.judge_logs, [])
        self.assertEqual(state.used_facts, [])

    def test_render_transcript_markdown_contains_dialogue_and_judge_logs(self) -> None:
        state = GameState(
            current_hp=85,
            player_hp=100,
            chat_history=[
                {"role": "assistant", "content": "Opening claim."},
                {"role": "user", "content": "Counterargument."},
            ],
            judge_logs=[
                {"turn": 1, "verdict": {"damage": 15, "reasoning": "Strong move."}},
            ],
        )

        transcript = render_transcript_markdown(state)

        self.assertIn("## Dialogue", transcript)
        self.assertIn("### Boss", transcript)
        self.assertIn("Counterargument.", transcript)
        self.assertIn("## Judge Logs", transcript)
        self.assertIn("Strong move.", transcript)

    def test_save_history_snapshot_writes_game_json_and_transcript(self) -> None:
        state = GameState(
            locale="en",
            campaign_id="earth_shape",
            stage_index=0,
            current_hp=90,
            player_hp=100,
            chat_history=[{"role": "user", "content": "Test argument."}],
            judge_logs=[
                {"turn": 1, "verdict": {"damage": 15, "player_damage": 0, "reasoning": "Strong move."}},
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = save_history_snapshot(
                Path(tmpdir),
                state,
                status="paused",
                metadata={
                    "campaign_title": "Earth",
                    "stage_title": "Flat Earth",
                    "boss_name": "Anaximenes",
                    "llm_mode": "local",
                },
            )

            game_json = json.loads((session_dir / "game.json").read_text(encoding="utf-8"))
            transcript = (session_dir / "transcript.md").read_text(encoding="utf-8")

            self.assertEqual(game_json["status"], "paused")
            self.assertEqual(game_json["metadata"]["stage_title"], "Flat Earth")
            self.assertEqual(game_json["state"]["session_id"], state.session_id)
            self.assertIn("## Turn Summary", transcript)
            self.assertIn("- Status: paused", transcript)
            self.assertIn("- Stage title: Flat Earth", transcript)
            self.assertIn("Test argument.", transcript)


if __name__ == "__main__":
    unittest.main()
