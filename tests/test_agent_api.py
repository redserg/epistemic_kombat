import unittest

from agent_api import window_boss_history


class WindowBossHistoryTests(unittest.TestCase):
    def test_keeps_short_history_unchanged(self) -> None:
        history = [{"role": "assistant", "content": "greeting"}]
        self.assertEqual(window_boss_history(history, max_messages=4), history)

    def test_keeps_opening_greeting_and_latest_tail(self) -> None:
        history = [
            {"role": "assistant", "content": "greeting"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]

        result = window_boss_history(history, max_messages=4)

        self.assertEqual(result[0]["content"], "greeting")
        self.assertEqual([item["content"] for item in result[1:]], ["a1", "u2", "a2", "u3"][-3:])


if __name__ == "__main__":
    unittest.main()
