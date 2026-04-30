"""LLM mode and client configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

import yaml
from pydantic import BaseModel, Field

from agent_api import ModelConfig


class LLMModeConfig(BaseModel):
    label: str
    api_key_env: Optional[str] = None
    base_url_env: Optional[str] = None
    default_api_key: Optional[str] = None
    default_base_url: str
    model_aliases: Dict[str, str]
    judge_response_tokens: int = Field(default=220, ge=1)
    boss_response_tokens: int = Field(default=160, ge=1)


class LLMSettings(BaseModel):
    default_mode: str = "cloud"
    modes: Dict[str, LLMModeConfig]


class ResolvedLLMConfig(BaseModel):
    mode: str
    api_key: Optional[str] = None
    base_url: str
    role_models: Dict[str, ModelConfig]
    judge_response_tokens: int
    boss_response_tokens: int


def load_llm_settings(path: Path) -> LLMSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return LLMSettings.model_validate(raw)


def with_model_aliases(role_configs: Dict[str, ModelConfig], aliases: Dict[str, str]) -> Dict[str, ModelConfig]:
    resolved: Dict[str, ModelConfig] = {}
    for role_name, role_config in role_configs.items():
        resolved[role_name] = role_config.model_copy(update={"model": aliases.get(role_name, role_config.model)})
    return resolved


def resolve_llm_config(
    settings: LLMSettings,
    role_configs: Dict[str, ModelConfig],
    environ: Mapping[str, str] | None = None,
) -> ResolvedLLMConfig:
    env = environ or os.environ
    mode = env.get("LLM_MODE", settings.default_mode)
    if mode not in settings.modes:
        raise ValueError(f"Unknown LLM_MODE '{mode}'. Available modes: {', '.join(settings.modes)}")

    mode_cfg = settings.modes[mode]
    api_key = resolve_api_key(mode, mode_cfg, env)
    base_url = resolve_base_url(mode, mode_cfg, env)
    return ResolvedLLMConfig(
        mode=mode,
        api_key=api_key,
        base_url=base_url,
        role_models=with_model_aliases(role_configs, mode_cfg.model_aliases),
        judge_response_tokens=mode_cfg.judge_response_tokens,
        boss_response_tokens=mode_cfg.boss_response_tokens,
    )


def resolve_api_key(mode: str, mode_cfg: LLMModeConfig, env: Mapping[str, str]) -> Optional[str]:
    if "LLM_API_KEY" in env:
        return env["LLM_API_KEY"]

    if mode_cfg.api_key_env and mode_cfg.api_key_env in env:
        return env[mode_cfg.api_key_env]

    if mode == "cloud" and "NEBIUS_API_KEY" in env:
        return env["NEBIUS_API_KEY"]

    return mode_cfg.default_api_key


def resolve_base_url(mode: str, mode_cfg: LLMModeConfig, env: Mapping[str, str]) -> str:
    if "LLM_BASE_URL" in env:
        return env["LLM_BASE_URL"]

    if mode_cfg.base_url_env and mode_cfg.base_url_env in env:
        return env[mode_cfg.base_url_env]

    if mode == "cloud" and "NEBIUS_BASE_URL" in env:
        return env["NEBIUS_BASE_URL"]

    return mode_cfg.default_base_url


def validate_llm_config(resolved: ResolvedLLMConfig) -> None:
    if not resolved.base_url:
        raise ValueError("LLM base URL is required")

    if resolved.mode == "cloud" and not resolved.api_key:
        raise ValueError("Cloud mode requires TOKEN_FACTORY_API_KEY or LLM_API_KEY")

    missing_roles = [name for name, cfg in resolved.role_models.items() if not cfg.model]
    if missing_roles:
        raise ValueError(f"Missing model ids for roles: {', '.join(missing_roles)}")
