"""Console entrypoint for Epistemic Kombat."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
import yaml

# Load .env from project root before reading any env vars
load_dotenv(Path(__file__).parent / ".env")

from agent_api import AgentAPI, BossConfig, HistoricalFact, JudgeVerdict, ModelConfig
from state_manager import GameState, archive_game, clear_current, game_outcome, load_state, save_state

logger = logging.getLogger("epistemic_kombat")


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


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_boss_config(path: Path) -> BossConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return BossConfig.model_validate(raw)


def load_facts(path: Path) -> List[HistoricalFact]:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return [HistoricalFact.model_validate(item) for item in raw]


def load_model_config(path: Path) -> Dict[str, ModelConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {name: ModelConfig.model_validate(cfg) for name, cfg in raw.items()}


def main() -> None:
    root = Path(__file__).parent
    prompts_dir = root / "prompts"
    config_dir = root / "config"
    state_path = root / "state" / "current_game.json"
    history_root = root / "history"
    model_config_path = config_dir / "model_config.yaml"

    boss_prompt = load_prompt(prompts_dir / "boss_prompt.txt")
    judge_prompt = load_prompt(prompts_dir / "judge_prompt.txt")

    boss_config = load_boss_config(config_dir / "boss_config.yaml")
    facts = load_facts(config_dir / "historical_facts.json")
    model_cfg = load_model_config(model_config_path)

    judge_model_cfg = model_cfg.get("judge")
    boss_model_cfg = model_cfg.get("boss")

    if not judge_model_cfg or not boss_model_cfg:
        panel("Model configuration missing for judge or boss", title="Config error")
        sys.exit(1)

    initial_state = GameState(current_hp=boss_config.start_hp, player_hp=boss_config.player_start_hp)
    state = load_state(state_path, fallback=initial_state)

    # Create session directory and set up logging
    session_dir = history_root / state.session_id
    setup_logging(session_dir)
    logger.info("Session started: %s", state.session_id)

    # Nebius Token Factory only
    nebius_api_key = os.getenv("NEBIUS_API_KEY")
    nebius_base_url = os.getenv("NEBIUS_BASE_URL")

    if not nebius_api_key or not nebius_base_url:
        panel("NEBIUS_API_KEY and NEBIUS_BASE_URL are required", title="Config error")
        sys.exit(1)

    api = AgentAPI(api_key=nebius_api_key, base_url=nebius_base_url)

    cprint("🧠 Добро пожаловать в Epistemic Kombat! Введи 'quit', чтобы выйти.")
    panel(
        f"Босс: {boss_config.name}\nЭпоха: {boss_config.epoch}\nHP Босса: {state.current_hp} | HP Игрока: {state.player_hp}",
        title="Матч начался"
    )

    if state.turn_number == 1 and not state.chat_history:
        greeting = (
            "Приветствую тебя, искатель истины! Я Анаксимен из Милета. Задумывался ли ты когда-нибудь о дыхании, "
            "которое дает нам жизнь? Я утверждаю, что именно эта субстанция, аэр (воздух), является бесконечным "
            "источником всех вещей. Находясь в постоянном движении, он сгущается, превращаясь в ветер, воду, землю "
            "и даже твердый камень. Наш мир, мой друг, — это широкий плоский диск, образованный сжатием этого воздуха. "
            "Он надежно покоится на огромной, невидимой подушке из того же самого элемента. Солнце и луна подобны "
            "широким огненным листьям, парящим на ветрах над нами, а небеса вращаются вокруг нашей плоской Земли, "
            "как войлочная шапка на голове. Что скажешь об этой изящной, зримой механике природы?"
        )
        panel(greeting, title=f"{boss_config.name} ({state.current_hp} HP)")
        state.chat_history.append({"role": "assistant", "content": greeting})

    while True:
        outcome = game_outcome(state)

        if outcome == "victory":
            defeat_text = (
                "Я вынужден уступить! Твои умозаключения так же ясны и ярки, как тот разреженный воздух, что рождает огонь. "
                "Подумать только, Земля — это не плоский лист, парящий на ветрах, а великолепная сфера, подвешенная в космосе... "
                "это модель невероятной, головокружительной красоты. Ты показал мне, что истинная философия требует не только "
                "наблюдения за частями, но и понимания геометрии целого. Я благодарю тебя за это прозрение, друг. "
                "Пусть твой разум всегда остается таким же всеобъемлющим и безграничным, как сам воздух!"
            )
            panel(defeat_text, title="Победа! Босс повержен")
            break

        if outcome == "defeat":
            loss_text = (
                "Ты потерял доверие аудитории! Твои аргументы оказались слабы, полны анахронизмов и повторов. "
                "Анаксимен торжествующе поднимает руку: 'Вот видишь, друг мой? Плоский диск на подушке воздуха — "
                "это единственная истина, доступная ясному разуму!'"
            )
            panel(loss_text, title="Поражение! Ты изгнан из Академии")
            break

        turn_label = f"Ход {state.turn_number}"
        if Prompt:
            player_msg = Prompt.ask(f"{turn_label} - Твой аргумент")
        else:
            player_msg = input(f"{turn_label} - Твой аргумент: ")

        if player_msg.strip().lower() in {"quit", "exit", "выход"}:
            cprint("Покидаем арену.")
            break

        # Update history with player message
        state.chat_history.append({"role": "user", "content": player_msg})

        boss_state: Dict[str, int] = {
            "current_hp": state.current_hp,
            "player_hp": state.player_hp,
            "turn_number": state.turn_number,
        }

        # Judge evaluates player's move
        try:
            verdict: JudgeVerdict = api.judge(
                model=judge_model_cfg,
                system_prompt=judge_prompt,
                player_message=player_msg,
                boss_state=boss_state,
                facts=facts,
                used_facts=state.used_facts,
            )
        except Exception as exc:  # noqa: BLE001
            panel(f"Judge failed: {exc}", title="Error")
            verdict = JudgeVerdict(is_anachronism=False, damage=0, player_damage=0, reasoning="Fallback: judge error", hidden_directive="Hold stance")

        damage = max(0, min(20, verdict.damage))
        player_damage = max(0, min(20, verdict.player_damage))
        state.current_hp = max(0, state.current_hp - damage)
        state.player_hp = max(0, state.player_hp - player_damage)

        # Track used facts
        if verdict.used_fact_summary:
            state.used_facts.append(verdict.used_fact_summary)

        state.judge_logs.append(
            {
                "turn": state.turn_number,
                "verdict": verdict.model_dump(),
            }
        )

        panel(
            f"{'⚠ Анахронизм!' if verdict.is_anachronism else 'Эпоха ОК'} | Урон боссу: {damage} | Урон игроку: {player_damage}\nРешение: {verdict.reasoning}",
            title="Судья"
        )

        # Boss responds with hidden directive
        try:
            boss_reply = api.boss(
                model=boss_model_cfg,
                system_prompt=boss_prompt,
                chat_history=state.chat_history,
                hidden_directive=verdict.hidden_directive,
            )
        except Exception as exc:  # noqa: BLE001
            boss_reply = f"[System] Boss failed to respond: {exc}"

        state.chat_history.append({"role": "assistant", "content": boss_reply})

        panel(boss_reply, title=f"{boss_config.name} (Boss HP: {state.current_hp} | Player HP: {state.player_hp})")

        # Persist after each full turn
        save_state(state_path, state)

        state.turn_number += 1

    save_state(state_path, state)

    if game_outcome(state):
        dest = archive_game(history_root, state)
        clear_current(state_path)
        logger.info("Game archived to %s", dest)
        cprint(f"Спасибо за игру. История сохранена в history/{state.session_id}/")
    else:
        logger.info("Game paused: %s", state.session_id)
        cprint(f"Игра приостановлена. Продолжение сохранено в state/current_game.json ({state.session_id}).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
