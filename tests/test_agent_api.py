import unittest

from agent_api import extract_json_object, limit_max_tokens, window_boss_history


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


class JudgeResponseHelperTests(unittest.TestCase):
    def test_extracts_json_from_wrapped_text(self) -> None:
        content = 'prefix {"damage": 5, "reasoning": "ok"} suffix'
        self.assertEqual(extract_json_object(content), '{"damage": 5, "reasoning": "ok"}')

    def test_handles_braces_inside_json_strings(self) -> None:
        content = '{"reasoning": "Use {ships} as evidence", "damage": 7}'
        self.assertEqual(extract_json_object(content), content)

    def test_clamps_existing_max_tokens(self) -> None:
        params = limit_max_tokens({"max_tokens": 700, "temperature": 0.2}, 220)
        self.assertEqual(params["max_tokens"], 220)
        self.assertEqual(params["temperature"], 0.2)

    def test_sets_max_tokens_when_missing(self) -> None:
        params = limit_max_tokens({"temperature": 0.2}, 220)
        self.assertEqual(params["max_tokens"], 220)


if __name__ == "__main__":
    unittest.main()
