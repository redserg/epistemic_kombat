from types import SimpleNamespace
import unittest

from agent_api import JudgeVerdict, ModelConfig
from main import parse_turn_command, run_self_check


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


class SelfCheckTests(unittest.TestCase):
    def test_parse_turn_command_understands_help_status_and_quit(self) -> None:
        self.assertEqual(parse_turn_command("help"), "help")
        self.assertEqual(parse_turn_command("/status"), "status")
        self.assertEqual(parse_turn_command("выход"), "quit")
        self.assertIsNone(parse_turn_command("argument text"))

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


if __name__ == "__main__":
    unittest.main()
