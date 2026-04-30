from pathlib import Path
import unittest

from game_content import load_campaign_catalog, render_prompt


class GameContentTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.catalog = load_campaign_catalog(root / "config" / "campaigns")
        self.root = root

    def test_loads_both_locales_and_campaigns(self) -> None:
        self.assertIn("ru", self.catalog.locales())
        self.assertIn("en", self.catalog.locales())
        self.assertIn("earth_shape", self.catalog.campaigns_for_locale("ru"))
        self.assertIn("light_nature", self.catalog.campaigns_for_locale("en"))
        self.assertIn("earth_history", self.catalog.campaigns_for_locale("ru"))
        self.assertIn("chemistry_revolutions", self.catalog.campaigns_for_locale("en"))
        self.assertEqual(len(self.catalog.campaigns_for_locale("ru")), 4)

    def test_earth_shape_campaign_has_multiple_stages(self) -> None:
        campaign = self.catalog.get("ru", "earth_shape")
        self.assertGreaterEqual(len(campaign.stages), 4)
        self.assertEqual(campaign.stages[0].boss_name, "Анаксимен Милетский")
        self.assertEqual(campaign.stages[0].start_hp, 90)

    def test_light_campaign_now_has_wave_stage(self) -> None:
        campaign = self.catalog.get("en", "light_nature")
        self.assertGreaterEqual(len(campaign.stages), 4)
        self.assertEqual(campaign.stages[2].id, "young")
        self.assertIn("Two slits produce stable interference fringes", campaign.stages[2].facts[0].fact)

    def test_new_campaigns_have_three_stages_per_locale(self) -> None:
        for locale in ("ru", "en"):
            self.assertEqual(len(self.catalog.get(locale, "earth_history").stages), 3)
            self.assertEqual(len(self.catalog.get(locale, "chemistry_revolutions").stages), 3)

    def test_english_flat_earth_stage_uses_tuned_hp(self) -> None:
        campaign = self.catalog.get("en", "earth_shape")
        self.assertEqual(campaign.stages[0].start_hp, 90)

    def test_duplicate_campaign_ids_across_split_files_raise(self) -> None:
        duplicate_dir = self.root / "tests" / "fixtures" / "duplicate_campaigns"

        with self.assertRaisesRegex(ValueError, "Duplicate campaign 'earth_shape'"):
            load_campaign_catalog(duplicate_dir)

    def test_prompt_render_includes_stage_specific_content(self) -> None:
        campaign = self.catalog.get("en", "light_nature")
        stage = campaign.stages[0]
        template = (self.root / "prompts" / "judge_prompt.en.txt").read_text(encoding="utf-8")
        rendered = render_prompt(template, campaign, stage)

        self.assertIn("Empedocles", rendered)
        self.assertIn("camera obscura", rendered)
        self.assertIn("The Nature of Light", rendered)

    def test_boss_prompt_contains_non_concession_rule(self) -> None:
        campaign = self.catalog.get("en", "earth_shape")
        stage = campaign.stages[0]
        template = (self.root / "prompts" / "boss_prompt.en.txt").read_text(encoding="utf-8")
        rendered = render_prompt(template, campaign, stage)

        self.assertIn("do not fully agree with the player's main thesis", rendered)

    def test_boss_prompt_contains_anti_anachronism_language_rule(self) -> None:
        campaign = self.catalog.get("ru", "light_nature")
        stage = campaign.stages[0]
        template = (self.root / "prompts" / "boss_prompt.ru.txt").read_text(encoding="utf-8")
        rendered = render_prompt(template, campaign, stage)

        self.assertIn("Не употребляй названия будущих экспериментов", rendered)


if __name__ == "__main__":
    unittest.main()
