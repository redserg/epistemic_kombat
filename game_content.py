"""Campaign and stage content loading for Epistemic Kombat."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field

from agent_api import HistoricalFact


class StageConfig(BaseModel):
    id: str
    title: str
    topic: str
    boss_name: str
    epoch: str
    worldview: List[str] = Field(default_factory=list)
    boss_voice: List[str] = Field(default_factory=list)
    judge_focus: List[str] = Field(default_factory=list)
    principles: List[str] = Field(default_factory=list)
    facts: List[HistoricalFact] = Field(default_factory=list)
    greeting: str
    victory_text: str
    defeat_text: str
    stage_clear_text: str
    start_hp: int = 100
    player_start_hp: int = 100
    response_language: str


class CampaignConfig(BaseModel):
    id: str
    locale: str
    title: str
    description: str
    introduction: str
    completion_text: str
    stages: List[StageConfig]


class CampaignCatalog(BaseModel):
    campaigns: Dict[str, Dict[str, CampaignConfig]]

    def locales(self) -> List[str]:
        return list(self.campaigns)

    def campaigns_for_locale(self, locale: str) -> Dict[str, CampaignConfig]:
        return self.campaigns[locale]

    def get(self, locale: str, campaign_id: str) -> CampaignConfig:
        return self.campaigns[locale][campaign_id]


def load_campaign_catalog(path: Path) -> CampaignCatalog:
    if path.is_dir():
        raw = {"campaigns": {}}
        for campaign_path in sorted(path.glob("*.yaml")):
            payload = yaml.safe_load(campaign_path.read_text(encoding="utf-8")) or {}
            for locale, campaigns in (payload.get("campaigns") or {}).items():
                locale_bucket = raw["campaigns"].setdefault(locale, {})
                for campaign_id, campaign in campaigns.items():
                    if campaign_id in locale_bucket:
                        raise ValueError(
                            f"Duplicate campaign '{campaign_id}' for locale '{locale}' in {campaign_path}"
                        )
                    locale_bucket[campaign_id] = campaign
    else:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CampaignCatalog.model_validate(raw)


def format_lines(lines: List[str]) -> str:
    """Render a list of strings as compact bullet text for prompts."""
    if not lines:
        return "- none"
    return "\n".join(f"- {line}" for line in lines)


def render_prompt(template: str, campaign: CampaignConfig, stage: StageConfig) -> str:
    """Substitute stage content into a localized prompt template."""
    values = {
        "campaign_title": campaign.title,
        "campaign_description": campaign.description,
        "stage_title": stage.title,
        "topic": stage.topic,
        "boss_name": stage.boss_name,
        "epoch": stage.epoch,
        "worldview": format_lines(stage.worldview),
        "boss_voice": format_lines(stage.boss_voice),
        "judge_focus": format_lines(stage.judge_focus),
        "principles": format_lines(stage.principles),
        "facts": format_lines(
            [
                f"{fact.fact} (source: {fact.source})" if fact.source else fact.fact
                for fact in stage.facts
            ]
        ),
        "response_language": stage.response_language,
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered
