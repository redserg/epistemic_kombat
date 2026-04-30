"""Console entrypoint for Epistemic Kombat."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
import yaml

# Load .env from project root before reading any env vars
load_dotenv(Path(__file__).parent / ".env")

from agent_api import AgentAPI, JudgeVerdict, ModelConfig
from game_content import CampaignCatalog, CampaignConfig, StageConfig, load_campaign_catalog, render_prompt
from state_manager import (
    GameState,
    advance_turn,
    archive_game,
    clear_current,
    game_outcome,
    load_state,
    reset_stage_progress,
    save_state,
)

logger = logging.getLogger("epistemic_kombat")

UI_TEXT: Dict[str, Dict[str, str]] = {
    "ru": {
        "welcome": "🧠 Добро пожаловать в Epistemic Kombat! Введи 'quit', чтобы выйти.",
        "config_error": "Ошибка конфигурации",
        "language_prompt": "Language / Язык",
        "campaign_prompt": "Выбери кампанию",
        "resume_prompt": "Найдено сохранение. Продолжить (`resume`) или начать заново (`new`)?",
        "match_started": "Матч начался",
        "judge_title": "Судья",
        "victory_title": "Победа! Уровень пройден",
        "defeat_title": "Поражение",
        "campaign_complete_title": "Кампания завершена",
        "stage_clear_title": "Новый уровень открыт",
        "paused": "Игра приостановлена. Продолжение сохранено в state/current_game.json ({session_id}).",
        "archived": "Спасибо за игру. История сохранена в history/{session_id}/",
        "quit": "Покидаем арену.",
        "turn_label": "Ход {turn_number}",
        "argument_prompt": "Твой аргумент",
        "saved_state_missing": "Сохранённая игра ссылается на неизвестную кампанию. Начинаю новую.",
        "resume_choices": "resume,new",
        "stage_header": "Кампания: {campaign_title}\nУровень: {stage_title}\nБосс: {boss_name}\nЭпоха: {epoch}\nТема: {topic}\nHP Босса: {boss_hp} | HP Игрока: {player_hp}",
        "boss_status": "{boss_name} (Boss HP: {boss_hp} | Player HP: {player_hp})",
        "judge_result": "{status} | Урон боссу: {damage} | Урон игроку: {player_damage}\nРешение: {reasoning}",
        "anachronism": "⚠ Анахронизм!",
        "valid_epoch": "Эпоха ОК",
        "system_boss_fail": "[Система] Босс не смог ответить: {error}",
        "system_judge_fail": "Judge failed: {error}",
    },
    "en": {
        "welcome": "🧠 Welcome to Epistemic Kombat! Type 'quit' to leave.",
        "config_error": "Config error",
        "language_prompt": "Language / Язык",
        "campaign_prompt": "Choose a campaign",
        "resume_prompt": "Saved progress found. Continue (`resume`) or start over (`new`)?",
        "match_started": "Match Started",
        "judge_title": "Judge",
        "victory_title": "Victory! Stage Cleared",
        "defeat_title": "Defeat",
        "campaign_complete_title": "Campaign Complete",
        "stage_clear_title": "Next Stage Unlocked",
        "paused": "Game paused. Progress saved in state/current_game.json ({session_id}).",
        "archived": "Thanks for playing. History saved in history/{session_id}/",
        "quit": "Leaving the arena.",
        "turn_label": "Turn {turn_number}",
        "argument_prompt": "Your argument",
        "saved_state_missing": "Saved game points to an unknown campaign. Starting a new one.",
        "resume_choices": "resume,new",
        "stage_header": "Campaign: {campaign_title}\nStage: {stage_title}\nBoss: {boss_name}\nEra: {epoch}\nTopic: {topic}\nBoss HP: {boss_hp} | Player HP: {player_hp}",
        "boss_status": "{boss_name} (Boss HP: {boss_hp} | Player HP: {player_hp})",
        "judge_result": "{status} | Boss damage: {damage} | Player damage: {player_damage}\nRuling: {reasoning}",
        "anachronism": "⚠ Anachronism!",
        "valid_epoch": "Era OK",
        "system_boss_fail": "[System] Boss failed to respond: {error}",
        "system_judge_fail": "Judge failed: {error}",
    },
}


def setup_logging(session_dir: Path) -> None:
    """Configure logging into history/<session_id>/game.log + stderr."""
    session_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(session_dir / "game.log", encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )


try:  # Optional pretty console
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()
except Exception:  # noqa: BLE001
    console = None
    Prompt = None  # type: ignore


def cprint(text: str) -> None:
    if console:
        console.print(text)
    else:
        print(text)


def panel(text: str, title: str | None = None) -> None:
    if console:
        console.print(Panel(text, title=title, expand=False))
    else:
        if title:
            print(f"[{title}] {text}")
        else:
            print(text)


def text(locale: str, key: str, **kwargs: object) -> str:
    template = UI_TEXT[locale][key]
    return template.format(**kwargs)


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_model_config(path: Path) -> Dict[str, ModelConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {name: ModelConfig.model_validate(cfg) for name, cfg in raw.items()}


def ask(prompt_text: str, *, choices: list[str] | None = None, default: str | None = None) -> str:
    if Prompt:
        prompt_kwargs: Dict[str, object] = {}
        if choices:
            prompt_kwargs["choices"] = choices
        if default:
            prompt_kwargs["default"] = default
        return Prompt.ask(prompt_text, **prompt_kwargs)

    suffix = f" [{'/'.join(choices)}]" if choices else ""
    if default:
        suffix = f"{suffix} (default: {default})"
    answer = input(f"{prompt_text}{suffix}: ").strip()
    if not answer and default is not None:
        return default
    return answer


def choose_locale(catalog: CampaignCatalog) -> str:
    return ask(
        text("ru", "language_prompt"),
        choices=catalog.locales(),
        default="ru",
    )


def choose_campaign(catalog: CampaignCatalog, locale: str) -> CampaignConfig:
    available = list(catalog.campaigns_for_locale(locale).items())
    for index, (_, campaign) in enumerate(available, start=1):
        cprint(f"{index}. {campaign.title} — {campaign.description}")

    choice = ask(
        text(locale, "campaign_prompt"),
        choices=[str(index) for index in range(1, len(available) + 1)],
        default="1",
    )
    _, campaign = available[int(choice) - 1]
    return campaign


def create_new_state(catalog: CampaignCatalog) -> GameState:
    locale = choose_locale(catalog)
    campaign = choose_campaign(catalog, locale)
    panel(campaign.introduction, title=campaign.title)
    first_stage = campaign.stages[0]
    return GameState(
        locale=locale,
        campaign_id=campaign.id,
        stage_index=0,
        current_hp=first_stage.start_hp,
        player_hp=first_stage.player_start_hp,
    )


def resolve_campaign(catalog: CampaignCatalog, state: GameState) -> CampaignConfig | None:
    try:
        return catalog.get(state.locale, state.campaign_id)
    except KeyError:
        return None


def get_stage(campaign: CampaignConfig, state: GameState) -> StageConfig:
    return campaign.stages[state.stage_index]


def snapshot_stage(state: GameState, campaign: CampaignConfig, stage: StageConfig, outcome: str) -> Dict[str, object]:
    return {
        "campaign_id": campaign.id,
        "campaign_title": campaign.title,
        "stage_id": stage.id,
        "stage_title": stage.title,
        "outcome": outcome,
        "boss_hp": state.current_hp,
        "player_hp": state.player_hp,
        "turn_number": state.turn_number,
        "used_facts": list(state.used_facts),
        "judge_logs": list(state.judge_logs),
        "chat_history": list(state.chat_history),
    }


def show_stage_header(locale: str, campaign: CampaignConfig, stage: StageConfig, state: GameState) -> None:
    panel(
        text(
            locale,
            "stage_header",
            campaign_title=campaign.title,
            stage_title=stage.title,
            boss_name=stage.boss_name,
            epoch=stage.epoch,
            topic=stage.topic,
            boss_hp=state.current_hp,
            player_hp=state.player_hp,
        ),
        title=text(locale, "match_started"),
    )


def ensure_stage_greeting(stage: StageConfig, state: GameState) -> None:
    if state.chat_history:
        return
    panel(stage.greeting, title=f"{stage.boss_name} ({state.current_hp} HP)")
    state.chat_history.append({"role": "assistant", "content": stage.greeting})


def judge_status_text(locale: str, verdict: JudgeVerdict, damage: int, player_damage: int) -> str:
    status = text(locale, "anachronism") if verdict.is_anachronism else text(locale, "valid_epoch")
    return text(
        locale,
        "judge_result",
        status=status,
        damage=damage,
        player_damage=player_damage,
        reasoning=verdict.reasoning,
    )


def main() -> None:
    root = Path(__file__).parent
    prompts_dir = root / "prompts"
    config_dir = root / "config"
    state_path = root / "state" / "current_game.json"
    history_root = root / "history"

    catalog = load_campaign_catalog(config_dir / "campaigns.yaml")
    model_cfg = load_model_config(config_dir / "model_config.yaml")

    judge_model_cfg = model_cfg.get("judge")
    boss_model_cfg = model_cfg.get("boss")
    if not judge_model_cfg or not boss_model_cfg:
        panel("Model configuration missing for judge or boss", title=text("ru", "config_error"))
        sys.exit(1)

    saved_state = load_state(state_path)
    state: GameState
    campaign: CampaignConfig

    if saved_state:
        saved_locale = saved_state.locale if saved_state.locale in catalog.locales() else "ru"
        campaign = resolve_campaign(catalog, saved_state)
        if campaign and ask(
            text(saved_locale, "resume_prompt"),
            choices=text(saved_locale, "resume_choices").split(","),
            default="resume",
        ) == "resume":
            state = saved_state
        else:
            if not campaign:
                panel(text(saved_locale, "saved_state_missing"), title=text(saved_locale, "config_error"))
            clear_current(state_path)
            state = create_new_state(catalog)
            campaign = catalog.get(state.locale, state.campaign_id)
    else:
        state = create_new_state(catalog)
        campaign = catalog.get(state.locale, state.campaign_id)

    session_dir = history_root / state.session_id
    setup_logging(session_dir)
    logger.info("Session started: %s", state.session_id)

    nebius_api_key = os.getenv("NEBIUS_API_KEY")
    nebius_base_url = os.getenv("NEBIUS_BASE_URL")
    if not nebius_api_key or not nebius_base_url:
        panel("NEBIUS_API_KEY and NEBIUS_BASE_URL are required", title=text(state.locale, "config_error"))
        sys.exit(1)

    api = AgentAPI(api_key=nebius_api_key, base_url=nebius_base_url)
    cprint(text(state.locale, "welcome"))
    show_stage_header(state.locale, campaign, get_stage(campaign, state), state)

    while True:
        campaign = catalog.get(state.locale, state.campaign_id)
        stage = get_stage(campaign, state)
        ensure_stage_greeting(stage, state)
        outcome = game_outcome(state)

        if outcome == "victory":
            panel(stage.victory_text, title=text(state.locale, "victory_title"))
            state.completed_stages.append(snapshot_stage(state, campaign, stage, outcome))

            if state.stage_index < len(campaign.stages) - 1:
                panel(stage.stage_clear_text, title=text(state.locale, "stage_clear_title"))
                state.stage_index += 1
                next_stage = get_stage(campaign, state)
                reset_stage_progress(
                    state,
                    boss_hp=next_stage.start_hp,
                    player_hp=next_stage.player_start_hp,
                )
                save_state(state_path, state)
                show_stage_header(state.locale, campaign, next_stage, state)
                continue

            panel(campaign.completion_text, title=text(state.locale, "campaign_complete_title"))
            break

        if outcome == "defeat":
            panel(stage.defeat_text, title=text(state.locale, "defeat_title"))
            state.completed_stages.append(snapshot_stage(state, campaign, stage, outcome))
            break

        turn_label = text(state.locale, "turn_label", turn_number=state.turn_number)
        player_msg = ask(f"{turn_label} - {text(state.locale, 'argument_prompt')}")

        if player_msg.strip().lower() in {"quit", "exit", "выход"}:
            cprint(text(state.locale, "quit"))
            break

        state.chat_history.append({"role": "user", "content": player_msg})
        boss_state: Dict[str, object] = {
            "current_hp": state.current_hp,
            "player_hp": state.player_hp,
            "turn_number": state.turn_number,
            "campaign_id": state.campaign_id,
            "stage_index": state.stage_index,
            "topic": stage.topic,
        }

        judge_prompt = render_prompt(
            load_prompt(prompts_dir / f"judge_prompt.{state.locale}.txt"),
            campaign,
            stage,
        )
        boss_prompt = render_prompt(
            load_prompt(prompts_dir / f"boss_prompt.{state.locale}.txt"),
            campaign,
            stage,
        )

        try:
            verdict = api.judge(
                model=judge_model_cfg,
                system_prompt=judge_prompt,
                player_message=player_msg,
                boss_state=boss_state,
                facts=stage.facts,
                locale=state.locale,
                used_facts=state.used_facts,
            )
        except Exception as exc:  # noqa: BLE001
            panel(
                text(state.locale, "system_judge_fail", error=exc),
                title=text(state.locale, "config_error"),
            )
            verdict = JudgeVerdict(
                is_anachronism=False,
                damage=0,
                player_damage=0,
                reasoning="Fallback: judge error",
                hidden_directive="Hold stance",
            )

        damage = max(0, min(20, verdict.damage))
        player_damage = max(0, min(20, verdict.player_damage))
        state.current_hp = max(0, state.current_hp - damage)
        state.player_hp = max(0, state.player_hp - player_damage)

        if verdict.used_fact_summary:
            state.used_facts.append(verdict.used_fact_summary)

        state.judge_logs.append({"turn": state.turn_number, "verdict": verdict.model_dump()})
        panel(
            judge_status_text(state.locale, verdict, damage, player_damage),
            title=text(state.locale, "judge_title"),
        )

        try:
            boss_reply = api.boss(
                model=boss_model_cfg,
                system_prompt=boss_prompt,
                chat_history=state.chat_history,
                hidden_directive=verdict.hidden_directive,
                locale=state.locale,
            )
        except Exception as exc:  # noqa: BLE001
            boss_reply = text(state.locale, "system_boss_fail", error=exc)

        state.chat_history.append({"role": "assistant", "content": boss_reply})
        panel(
            boss_reply,
            title=text(
                state.locale,
                "boss_status",
                boss_name=stage.boss_name,
                boss_hp=state.current_hp,
                player_hp=state.player_hp,
            ),
        )

        advance_turn(state)
        save_state(state_path, state)

    save_state(state_path, state)
    if game_outcome(state):
        dest = archive_game(history_root, state)
        clear_current(state_path)
        logger.info("Game archived to %s", dest)
        cprint(text(state.locale, "archived", session_id=state.session_id))
    else:
        logger.info("Game paused: %s", state.session_id)
        cprint(text(state.locale, "paused", session_id=state.session_id))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
