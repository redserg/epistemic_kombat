from pathlib import Path
import unittest

from agent_api import ModelConfig
from llm_config import load_llm_settings, resolve_llm_config, validate_llm_config


class LLMConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.settings = load_llm_settings(root / "config" / "llm_modes.yaml")
        self.role_configs = {
            "judge": ModelConfig(model="placeholder", temperature=0.2),
            "boss": ModelConfig(model="placeholder", temperature=0.9),
        }

    def test_cloud_mode_uses_token_factory_aliases(self) -> None:
        resolved = resolve_llm_config(
            self.settings,
            self.role_configs,
            {"LLM_MODE": "cloud", "TOKEN_FACTORY_API_KEY": "secret"},
        )

        self.assertEqual(resolved.mode, "cloud")
        self.assertEqual(resolved.base_url, "https://api.tokenfactory.nebius.com/v1/")
        self.assertEqual(resolved.api_key, "secret")
        self.assertEqual(resolved.role_models["judge"].model, "openai/gpt-oss-20b")
        self.assertEqual(resolved.role_models["boss"].model, "openai/gpt-oss-20b")

    def test_local_mode_uses_ollama_defaults(self) -> None:
        resolved = resolve_llm_config(
            self.settings,
            self.role_configs,
            {"LLM_MODE": "local"},
        )

        self.assertEqual(resolved.mode, "local")
        self.assertEqual(resolved.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(resolved.api_key, "ollama")
        self.assertEqual(resolved.role_models["judge"].model, "gpt-oss:20b")

    def test_global_overrides_win_over_mode_defaults(self) -> None:
        resolved = resolve_llm_config(
            self.settings,
            self.role_configs,
            {
                "LLM_MODE": "local",
                "LLM_API_KEY": "override-key",
                "LLM_BASE_URL": "http://localhost:9999/v1",
            },
        )

        self.assertEqual(resolved.api_key, "override-key")
        self.assertEqual(resolved.base_url, "http://localhost:9999/v1")

    def test_legacy_nebius_variables_still_work_for_cloud_mode(self) -> None:
        resolved = resolve_llm_config(
            self.settings,
            self.role_configs,
            {
                "LLM_MODE": "cloud",
                "NEBIUS_API_KEY": "legacy-secret",
                "NEBIUS_BASE_URL": "https://legacy.example/v1",
            },
        )

        self.assertEqual(resolved.api_key, "legacy-secret")
        self.assertEqual(resolved.base_url, "https://legacy.example/v1")

    def test_validate_llm_config_accepts_valid_resolution(self) -> None:
        resolved = resolve_llm_config(
            self.settings,
            self.role_configs,
            {"LLM_MODE": "local"},
        )
        validate_llm_config(resolved)

    def test_validate_llm_config_rejects_cloud_without_api_key(self) -> None:
        resolved = resolve_llm_config(
            self.settings,
            self.role_configs,
            {"LLM_MODE": "cloud"},
        )
        with self.assertRaisesRegex(ValueError, "Cloud mode requires"):
            validate_llm_config(resolved)


if __name__ == "__main__":
    unittest.main()
