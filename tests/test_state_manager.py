import unittest

from state_manager import GameState, game_outcome


class GameOutcomeTests(unittest.TestCase):
    def test_returns_victory_when_boss_hp_is_zero(self) -> None:
        state = GameState(current_hp=0, player_hp=42)
        self.assertEqual(game_outcome(state), "victory")

    def test_returns_defeat_when_player_hp_is_zero(self) -> None:
        state = GameState(current_hp=13, player_hp=0)
        self.assertEqual(game_outcome(state), "defeat")

    def test_returns_none_for_in_progress_game(self) -> None:
        state = GameState(current_hp=13, player_hp=42)
        self.assertIsNone(game_outcome(state))


if __name__ == "__main__":
    unittest.main()
