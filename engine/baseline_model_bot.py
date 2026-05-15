"""Engine adapter for basemodel checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from engine.player_interface import Bot


EngineAction = Tuple[str, Optional[int]]

FORBIDDEN_STATE_KEYS = {
    "all_hole_cards",
    "deck",
    "evaluator",
    "future_board_cards",
    "opponent_hole_cards",
    "player_hole_cards",
    "remaining_deck",
    "showdown_cards_by_player",
    "training_label",
    "winner",
}

ALLOWED_STATE_KEYS = {
    "active_players",
    "blinds",
    "board_cards",
    "call_amount",
    "field_pct_remaining",
    "hand_id",
    "hero_equity",
    "hero_equity_source",
    "hole_cards",
    "icm_pressure",
    "itm_distance",
    "last_action_amount",
    "min_raise",
    "opp_last_action_id",
    "opponent_range_pct",
    "opponent_stacks",
    "player_id",
    "position",
    "pot_size",
    "stack_size",
    "table_id",
    "tournament_id",
    "bb_per_avg_stack",
}


def add_basemodel_to_path(basemodel_root: str | Path) -> Path:
    root = Path(basemodel_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def resolve_checkpoint_path(checkpoint_path: str | Path, *, basemodel_root: str | Path, engine_root: str | Path | None = None) -> Path:
    path = Path(checkpoint_path).expanduser()
    if path.is_absolute():
        return path

    candidates = [Path(basemodel_root).expanduser().resolve() / path]
    if engine_root is not None:
        candidates.append(Path(engine_root).expanduser().resolve() / path)
    candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_checkpoint_model(checkpoint_path: str | Path, *, basemodel_root: str | Path, map_location: str = "cpu"):
    add_basemodel_to_path(basemodel_root)
    from poker_ai.checkpoint import load_checkpoint
    from poker_ai.model import PokerActorCritic

    model = PokerActorCritic()
    load_checkpoint(checkpoint_path, model, map_location=map_location)
    if hasattr(model, "eval"):
        model.eval()
    return model


def has_forbidden_state_keys(game_state: Dict[str, Any]) -> bool:
    keys = set(game_state)
    if keys.intersection(FORBIDDEN_STATE_KEYS):
        return True
    return any("hole" in key.lower() and key != "hole_cards" for key in keys)


def unexpected_state_keys(game_state: Dict[str, Any]) -> list[str]:
    return sorted(set(game_state) - ALLOWED_STATE_KEYS)


def is_engine_action(action: Any) -> bool:
    if not isinstance(action, tuple) or len(action) != 2:
        return False
    action_type, amount = action
    if action_type not in {"fold", "call", "raise"}:
        return False
    if action_type in {"fold", "call"}:
        return amount == 0
    return isinstance(amount, int) and amount > 0


def conservative_engine_action(game_state: Dict[str, Any]) -> EngineAction:
    try:
        add_basemodel_to_path(Path(__file__).resolve().parents[2] / "poker-ai-basemodel")
        from poker_ai.actions import conservative_action

        action = conservative_action(game_state)
        if is_engine_action(action):
            return action
    except Exception:
        pass
    if int(game_state.get("call_amount", 0) or 0) == 0:
        return ("call", 0)
    return ("call", 0) if int(game_state.get("stack_size", 0) or 0) > 0 else ("fold", 0)


class BaselineModelEngineBot(Bot):
    """Engine-compatible wrapper around the basemodel neural bot."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        basemodel_root: str | Path = "../poker-ai-basemodel",
        name: str = "BaselineModelEngineBot",
        deterministic: bool = True,
        decision_timeout_ms: int | None = None,
        equity_source: str = "treys",
        equity_fallback_source: str | None = "constant",
        equity_iterations: int | None = None,
        model=None,
        require_checkpoint: bool = True,
    ):
        super().__init__(name)
        self.basemodel_root = add_basemodel_to_path(basemodel_root)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.load_error = ""

        try:
            from poker_ai.bot import NeuralBaselineBot
            from poker_ai.equity import EquityConfig
        except Exception as exc:
            self.load_error = f"basemodel import failed: {exc}"
            if require_checkpoint:
                raise RuntimeError(self.load_error) from exc
            self._delegate = None
            return

        if model is None and checkpoint_path:
            try:
                model = load_checkpoint_model(checkpoint_path, basemodel_root=self.basemodel_root)
            except Exception as exc:
                self.load_error = f"checkpoint load failed: {exc}"
                if require_checkpoint:
                    raise RuntimeError(self.load_error) from exc
                model = None
        elif model is None and require_checkpoint:
            self.load_error = "checkpoint path is required"
            raise RuntimeError(self.load_error)

        equity_config = EquityConfig(
            source=equity_source,
            fallback_source=equity_fallback_source,
            iterations=equity_iterations,
        )
        self._delegate = NeuralBaselineBot(
            model=model,
            name=name,
            deterministic=deterministic,
            auto_create_model=False,
            decision_timeout_ms=decision_timeout_ms,
            equity_config=equity_config,
        )

    @property
    def fallback_counts(self) -> Dict[str, int]:
        if self._delegate is None:
            return {"no_model": 1, "inference_errors": 0, "illegal_actions": 0, "timeouts": 0}
        return dict(getattr(self._delegate, "fallback_counts", {}))

    @property
    def equity_counts(self) -> Dict[str, int]:
        if self._delegate is None:
            return {"computed": 0, "fallbacks": 0}
        return dict(getattr(self._delegate, "equity_counts", {}))

    def get_action(self, game_state: Dict[str, Any]) -> EngineAction:
        if has_forbidden_state_keys(game_state):
            return conservative_engine_action(game_state)
        if self._delegate is None:
            return conservative_engine_action(game_state)
        action = self._delegate.get_action(dict(game_state))
        if not is_engine_action(action):
            return conservative_engine_action(game_state)
        return action


def validate_visible_state(game_state: Dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if has_forbidden_state_keys(game_state):
        failures.append("state contains hidden-card or privileged keys")
    unexpected = unexpected_state_keys(game_state)
    if unexpected:
        failures.append(f"state contains unexpected keys: {', '.join(unexpected)}")
    return not failures, failures


def validate_engine_actions(actions: Iterable[EngineAction]) -> tuple[bool, list[str]]:
    failures = []
    for action in actions:
        if not is_engine_action(action):
            failures.append(f"invalid engine action: {action!r}")
    return not failures, failures
