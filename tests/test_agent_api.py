import unittest

from agent_api import (
    boss_reply_is_usable,
    boss_fallback_reply,
    boss_reply_reminder,
    boss_retry_reminder,
    ChatResult,
    clean_boss_reply,
    extract_json_object,
    judge_json_reminder,
    judge_retry_reminder,
    limit_max_tokens,
    window_boss_history,
)


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


class BossReplyHelperTests(unittest.TestCase):
    def test_collapses_multiline_reply_into_single_string(self) -> None:
        reply = "Я отвечаю.\n\nКоротко и ясно."
        self.assertEqual(clean_boss_reply(reply), "Я отвечаю. Коротко и ясно.")

    def test_returns_empty_string_for_blank_reply(self) -> None:
        self.assertEqual(clean_boss_reply("   \n\t"), "")

    def test_english_localization_helpers_are_english(self) -> None:
        self.assertIn("strictly valid JSON", judge_json_reminder("en"))
        self.assertIn("INVALID OR TRUNCATED".lower(), judge_retry_reminder("en").lower())
        self.assertIn("Direct speech only".lower(), boss_reply_reminder("en").lower())
        self.assertIn("EMPTY OR CUT OFF".lower(), boss_retry_reminder("en").lower())
        self.assertIn("Repeat your argument", boss_fallback_reply("en"))

    def test_rejects_non_stop_boss_reply(self) -> None:
        self.assertFalse(boss_reply_is_usable("Incomplete but non-empty", "length"))

    def test_rejects_reply_without_terminal_punctuation(self) -> None:
        self.assertFalse(boss_reply_is_usable("This still trails off", "stop"))

    def test_accepts_complete_boss_reply(self) -> None:
        self.assertTrue(boss_reply_is_usable("I still resist your claim.", "stop"))


if __name__ == "__main__":
    unittest.main()
