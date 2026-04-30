import unittest

from agent_api import (
    boss_generation_params_for_attempt,
    boss_response_token_limit_for_attempt,
    boss_reply_is_usable,
    boss_reply_is_fresh,
    boss_reply_matches_locale,
    boss_fallback_reply,
    boss_reply_reminder,
    boss_retry_reminder,
    ChatResult,
    clean_boss_reply,
    default_hidden_directive,
    extract_json_object,
    is_hidden_directive_safe,
    JudgeVerdict,
    judge_generation_params_for_attempt,
    judge_json_reminder,
    judge_response_token_limit_for_attempt,
    judge_response_format_for_attempt,
    judge_retry_reminder,
    latest_boss_reply,
    limit_max_tokens,
    HistoricalFact,
    message_matches_allowed_fact,
    normalize_false_anachronism,
    salvage_boss_reply,
    used_fact_summary_is_repeat,
    stabilize_hidden_directive,
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

    def test_local_judge_retry_disables_response_format(self) -> None:
        self.assertEqual(judge_response_format_for_attempt(llm_mode="cloud", attempt=2), {"type": "json_object"})
        self.assertIsNone(judge_response_format_for_attempt(llm_mode="local", attempt=2))

    def test_judge_retry_lowers_temperature(self) -> None:
        params = judge_generation_params_for_attempt({"temperature": 0.2}, 800, 2)
        self.assertEqual(params["max_tokens"], 260)
        self.assertEqual(params["temperature"], 0.1)

    def test_judge_retry_uses_stricter_token_cap(self) -> None:
        self.assertEqual(judge_response_token_limit_for_attempt(800, 1), 800)
        self.assertEqual(judge_response_token_limit_for_attempt(800, 2), 260)

    def test_repeat_fact_summary_detects_close_paraphrase(self) -> None:
        self.assertTrue(
            used_fact_summary_is_repeat(
                "Глаз не испускает свет, иначе бы видели в темноте.",
                ["Глаз не освещает предметы в полной темноте"],
            )
        )

    def test_normalized_verdict_downgrades_repeat_fact_summary(self) -> None:
        verdict = JudgeVerdict(
            is_anachronism=False,
            damage=15,
            player_damage=0,
            reasoning="Repeat presented as new.",
            hidden_directive="Admit the observation is powerful, but resist and defend your worldview.",
            used_fact_summary="Глаз не испускает свет, иначе бы видели в темноте.",
        ).normalized(
            used_facts=["Глаз не освещает предметы в полной темноте"],
            locale="ru",
        )
        self.assertEqual(verdict.damage, 4)
        self.assertEqual(verdict.player_damage, 3)
        self.assertEqual(verdict.used_fact_summary, "")

    def test_normalized_verdict_removes_player_damage_from_strong_valid_hit(self) -> None:
        verdict = JudgeVerdict(
            is_anachronism=False,
            damage=15,
            player_damage=5,
            reasoning="Strong move.",
            hidden_directive="Resist.",
            used_fact_summary="ships",
        ).normalized()
        self.assertEqual(verdict.player_damage, 0)

    def test_matches_allowed_fact_from_stage(self) -> None:
        facts = [HistoricalFact(fact="A small opening can project an inverted image of a bright scene onto a surface.")]
        self.assertTrue(
            message_matches_allowed_fact(
                "A small opening can project an inverted image of a bright scene, which makes sense if light travels from the object toward the observer.",
                facts,
            )
        )

    def test_normalize_false_anachronism_recovers_allowed_fact(self) -> None:
        verdict = JudgeVerdict(
            is_anachronism=True,
            damage=0,
            player_damage=12,
            reasoning="Camera obscura is anachronistic.",
            hidden_directive="Mock the anachronism and reject it sharply.",
            used_fact_summary="",
        )
        repaired = normalize_false_anachronism(
            verdict,
            player_message="A small opening can project an inverted image of a bright scene, which makes sense if light travels from the object toward the observer.",
            facts=[HistoricalFact(fact="A small opening can project an inverted image of a bright scene onto a surface.")],
            locale="en",
        )
        self.assertFalse(repaired.is_anachronism)
        self.assertEqual(repaired.player_damage, 0)
        self.assertGreaterEqual(repaired.damage, 8)


class BossReplyHelperTests(unittest.TestCase):
    def test_collapses_multiline_reply_into_single_string(self) -> None:
        reply = "Я отвечаю.\n\nКоротко и ясно."
        self.assertEqual(clean_boss_reply(reply), "Я отвечаю. Коротко и ясно.")

    def test_returns_empty_string_for_blank_reply(self) -> None:
        self.assertEqual(clean_boss_reply("   \n\t"), "")

    def test_latest_boss_reply_returns_last_assistant_line(self) -> None:
        history = [
            {"role": "assistant", "content": "First reply."},
            {"role": "user", "content": "Argument."},
            {"role": "assistant", "content": "Second reply."},
        ]
        self.assertEqual(latest_boss_reply(history), "Second reply.")

    def test_english_localization_helpers_are_english(self) -> None:
        self.assertIn("strictly valid JSON", judge_json_reminder("en"))
        self.assertIn("INVALID OR TRUNCATED".lower(), judge_retry_reminder("en").lower())
        self.assertIn("Direct speech only".lower(), boss_reply_reminder("en").lower())
        self.assertIn("EMPTY OR CUT OFF".lower(), boss_retry_reminder("en").lower())
        self.assertIn("defend my old view", boss_fallback_reply("en"))

    def test_rejects_non_stop_boss_reply(self) -> None:
        self.assertFalse(boss_reply_is_usable("Incomplete but non-empty", "length"))

    def test_rejects_reply_without_terminal_punctuation(self) -> None:
        self.assertFalse(boss_reply_is_usable("This still trails off", "stop"))

    def test_accepts_complete_boss_reply(self) -> None:
        self.assertTrue(boss_reply_is_usable("I still resist your claim.", "stop"))

    def test_rejects_near_duplicate_boss_reply(self) -> None:
        history = [
            {"role": "assistant", "content": "The eye's inner fire meets external light to create sight."},
        ]
        self.assertFalse(
            boss_reply_is_fresh(
                "The eye's inner fire still meets the external light to produce sight.",
                history,
            )
        )

    def test_accepts_fresh_boss_reply(self) -> None:
        history = [
            {"role": "assistant", "content": "The eye's inner fire meets external light to create sight."},
        ]
        self.assertTrue(
            boss_reply_is_fresh(
                "A bright ray may wound the eye because it overwhelms the fire within it.",
                history,
            )
        )

    def test_rejects_english_reply_in_russian_locale(self) -> None:
        self.assertFalse(boss_reply_is_usable("I still resist your claim.", "stop", locale="ru"))
        self.assertFalse(boss_reply_matches_locale("I still resist your claim.", "ru"))

    def test_rejects_cyrillic_leak_in_english_locale(self) -> None:
        self.assertFalse(boss_reply_is_usable("Я всё ещё спорю.", "stop", locale="en"))
        self.assertFalse(boss_reply_matches_locale("Я всё ещё спорю.", "en"))

    def test_rejects_overlong_local_boss_reply(self) -> None:
        reply = " ".join(["word"] * 56) + "."
        self.assertFalse(boss_reply_is_usable(reply, "stop", llm_mode="local"))

    def test_salvages_complete_sentences_from_truncated_local_reply(self) -> None:
        reply = (
            "I see your point, yet the horizon may still be shaped by the air above us. "
            "A flat Earth can still explain the sight. This unfinished tail"
        )
        self.assertEqual(
            salvage_boss_reply(reply, "length", llm_mode="local"),
            "I see your point, yet the horizon may still be shaped by the air above us. A flat Earth can still explain the sight.",
        )

    def test_salvages_short_clause_when_no_sentence_is_completed(self) -> None:
        reply = "I admit the bright light wounds the eye, yet the inner fire still seeks it and meets it in vision without surrendering"
        self.assertEqual(
            salvage_boss_reply(reply, "length", llm_mode="local"),
            "I admit the bright light wounds the eye, yet the inner fire still seeks it and meets it in vision without surrendering.",
        )

    def test_salvages_first_short_sentences_from_overlong_local_reply(self) -> None:
        reply = (
            "I grant the sight is striking, yet the air may veil the lower hull before the mast. "
            "A flat Earth can still explain that appearance. "
            + " ".join(["extra"] * 70)
            + "."
        )
        self.assertEqual(
            salvage_boss_reply(reply, "stop", llm_mode="local"),
            "I grant the sight is striking, yet the air may veil the lower hull before the mast. A flat Earth can still explain that appearance.",
        )

    def test_local_boss_reminder_is_stricter(self) -> None:
        self.assertIn("under 45 words", boss_reply_reminder("en", llm_mode="local").lower())
        self.assertIn("35 words", boss_retry_reminder("en", llm_mode="local").lower())
        self.assertIn(
            "different line of defense",
            boss_retry_reminder("en", llm_mode="local", previous_reply="Old reply").lower(),
        )

    def test_fallback_reply_is_in_character_instead_of_meta_retry(self) -> None:
        reply = boss_fallback_reply(
            "en",
            llm_mode="local",
            hidden_directive="Admit the observation is powerful, but resist and defend your worldview.",
        )
        self.assertNotIn("Repeat your argument", reply)
        self.assertIn("defend my old view", reply)

    def test_fallback_reply_avoids_repeating_latest_assistant_line(self) -> None:
        reply = boss_fallback_reply(
            "en",
            llm_mode="local",
            hidden_directive="Admit the observation is powerful, but resist and defend your worldview.",
            chat_history=[{"role": "assistant", "content": "Your point presses me, yet I still defend my old view. One hard observation does not force me to yield."}],
        )
        self.assertNotIn("One hard observation does not force me to yield", reply)

    def test_boss_retry_uses_stricter_token_cap(self) -> None:
        self.assertEqual(boss_response_token_limit_for_attempt(500, 1), 500)
        self.assertEqual(boss_response_token_limit_for_attempt(500, 2), 220)

    def test_boss_retry_lowers_temperature(self) -> None:
        params = boss_generation_params_for_attempt({"temperature": 0.9}, 220, 2)
        self.assertEqual(params["max_tokens"], 220)
        self.assertEqual(params["temperature"], 0.4)

    def test_stabilizes_english_accept_directive_before_boss_defeat(self) -> None:
        directive = stabilize_hidden_directive(
            "Accept the argument.",
            locale="en",
            remaining_boss_hp=85,
            damage=15,
            player_damage=0,
            is_anachronism=False,
        )
        self.assertIn("defend your worldview", directive.lower())

    def test_stabilizes_russian_accept_directive_before_boss_defeat(self) -> None:
        directive = stabilize_hidden_directive(
            "Согласись с аргументом игрока.",
            locale="ru",
            remaining_boss_hp=85,
            damage=15,
            player_damage=0,
            is_anachronism=False,
        )
        self.assertIn("продолжай", directive.lower())

    def test_keeps_directive_after_actual_victory(self) -> None:
        directive = stabilize_hidden_directive(
            "Accept the argument.",
            locale="en",
            remaining_boss_hp=0,
            damage=15,
            player_damage=0,
            is_anachronism=False,
        )
        self.assertEqual(directive, "Accept the argument.")

    def test_unsafe_directive_about_curvature_is_rewritten(self) -> None:
        directive = stabilize_hidden_directive(
            "Ответить аргументом о кривизне Земли.",
            locale="ru",
            remaining_boss_hp=85,
            damage=15,
            player_damage=5,
            is_anachronism=False,
        )
        self.assertNotIn("кривизне земли", directive.lower())
        self.assertIn("защищать", directive.lower())

    def test_safe_resistive_directive_is_preserved(self) -> None:
        directive = stabilize_hidden_directive(
            "Defend your worldview and resist the claim.",
            locale="en",
            remaining_boss_hp=85,
            damage=15,
            player_damage=0,
            is_anachronism=False,
        )
        self.assertEqual(directive, "Defend your worldview and resist the claim.")


if __name__ == "__main__":
    unittest.main()
