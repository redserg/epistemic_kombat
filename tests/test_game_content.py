from pathlib import Path
import unittest

from game_content import load_campaign_catalog, render_prompt


class GameContentTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.catalog = load_campaign_catalog(root / "config" / "campaigns.yaml")
        self.root = root

    def test_loads_both_locales_and_campaigns(self) -> None:
        self.assertIn("ru", self.catalog.locales())
        self.assertIn("en", self.catalog.locales())
        self.assertIn("earth_shape", self.catalog.campaigns_for_locale("ru"))
        self.assertIn("light_nature", self.catalog.campaigns_for_locale("en"))

    def test_earth_shape_campaign_has_multiple_stages(self) -> None:
        campaign = self.catalog.get("ru", "earth_shape")
        self.assertGreaterEqual(len(campaign.stages), 3)
        self.assertEqual(campaign.stages[0].boss_name, "Анаксимен Милетский")

    def test_prompt_render_includes_stage_specific_content(self) -> None:
        campaign = self.catalog.get("en", "light_nature")
        stage = campaign.stages[0]
        template = (self.root / "prompts" / "judge_prompt.en.txt").read_text(encoding="utf-8")
        rendered = render_prompt(template, campaign, stage)

        self.assertIn("Empedocles", rendered)
        self.assertIn("camera obscura", rendered)
        self.assertIn("The Nature of Light", rendered)


if __name__ == "__main__":
    unittest.main()
