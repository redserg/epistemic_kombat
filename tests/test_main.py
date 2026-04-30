from types import SimpleNamespace
import unittest

from agent_api import JudgeVerdict, ModelConfig
from main import parse_turn_command, player_argument_is_near_repeat, run_self_check


class FakeAgentAPI:
    def __init__(self) -> None:
        self.judge_calls = []
        self.boss_calls = []

    def judge(self, **kwargs):
        self.judge_calls.append(kwargs)
        return JudgeVerdict(
            is_anachronism=False,
            damage=12,
            player_damage=0,
            reasoning="Valid observation.",
            hidden_directive="Resist.",
            used_fact_summary="ships hull-first",
        )

    def boss(self, **kwargs):
        self.boss_calls.append(kwargs)
        return "I still resist your claim."


class BadFakeAgentAPI(FakeAgentAPI):
    def judge(self, **kwargs):
        return JudgeVerdict(
            is_anachronism=False,
            damage=0,
            player_damage=0,
            reasoning="",
            hidden_directive="Resist.",
            used_fact_summary="",
        )


class SelfCheckTests(unittest.TestCase):
    def test_parse_turn_command_understands_help_status_and_quit(self) -> None:
        self.assertEqual(parse_turn_command("help"), "help")
        self.assertEqual(parse_turn_command("/status"), "status")
        self.assertEqual(parse_turn_command("выход"), "quit")
        self.assertIsNone(parse_turn_command("argument text"))

    def test_player_argument_repeat_detects_close_paraphrase(self) -> None:
        history = [
            {"role": "user", "content": "Ships disappear hull-first below the horizon."},
        ]
        self.assertTrue(
            player_argument_is_near_repeat(
                "Ships disappear hull-first beneath the horizon.",
                history,
            )
        )

    def test_player_argument_repeat_allows_new_argument(self) -> None:
        history = [
            {"role": "user", "content": "Ships disappear hull-first below the horizon."},
        ]
        self.assertFalse(
            player_argument_is_near_repeat(
                "During a lunar eclipse the Earth casts a round shadow on the Moon.",
                history,
            )
        )

    def test_run_self_check_exercises_judge_and_boss(self) -> None:
        api = FakeAgentAPI()
        resolved_llm = SimpleNamespace(
            mode="local",
            base_url="http://127.0.0.1:11434/v1",
            judge_response_tokens=800,
            boss_response_tokens=500,
        )
        judge_cfg = ModelConfig(model="judge-model")
        boss_cfg = ModelConfig(model="boss-model")

        lines = run_self_check(
            api,
            judge_model_cfg=judge_cfg,
            boss_model_cfg=boss_cfg,
            resolved_llm=resolved_llm,
        )

        self.assertIn("LLM mode: local", lines[0])
        self.assertIn("Judge OK: damage=12", "\n".join(lines))
        self.assertIn("Boss OK: I still resist your claim.", "\n".join(lines))
        self.assertEqual(api.judge_calls[0]["llm_mode"], "local")
        self.assertEqual(api.boss_calls[0]["llm_mode"], "local")

    def test_run_self_check_rejects_suspicious_outputs(self) -> None:
        api = BadFakeAgentAPI()
        resolved_llm = SimpleNamespace(
            mode="local",
            base_url="http://127.0.0.1:11434/v1",
            judge_response_tokens=800,
            boss_response_tokens=500,
        )

        with self.assertRaisesRegex(ValueError, "Judge self-check looks suspicious"):
            run_self_check(
                api,
                judge_model_cfg=ModelConfig(model="judge-model"),
                boss_model_cfg=ModelConfig(model="boss-model"),
                resolved_llm=resolved_llm,
            )


if __name__ == "__main__":
    unittest.main()
