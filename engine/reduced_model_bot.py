"""Engine adapter for reduced-observation basemodel checkpoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from engine.baseline_model_bot import (
    EngineAction,
    add_basemodel_to_path,
    conservative_engine_action,
    has_forbidden_state_keys,
    is_engine_action,
    resolve_checkpoint_path,
)
from engine.player_interface import Bot


def load_reduced_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    basemodel_root: str | Path,
    observation_size: int = 8,
    map_location: str = "cpu",
):
    add_basemodel_to_path(basemodel_root)
    from poker_ai.checkpoint import load_checkpoint
    from poker_ai.model import PokerActorCritic

    model = PokerActorCritic(observation_size=observation_size)
    load_checkpoint(checkpoint_path, model, map_location=map_location)
    if hasattr(model, "eval"):
        model.eval()
    return model


class ReducedModelEngineBot(Bot):
    """Engine-compatible wrapper around the Phase 28 reduced-observation model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        basemodel_root: str | Path = "../poker-ai-basemodel",
        engine_root: str | Path | None = None,
        name: str = "ReducedModelEngineBot",
        deterministic: bool = True,
        decision_timeout_ms: int | None = None,
        equity_source: str = "pokerstove",
        equity_fallback_source: str | None = "constant",
        equity_iterations: int | None = None,
        observation_size: int = 8,
        require_checkpoint: bool = True,
    ):
        super().__init__(name)
        self.basemodel_root = add_basemodel_to_path(basemodel_root)
        self.checkpoint_path = resolve_checkpoint_path(
            checkpoint_path,
            basemodel_root=self.basemodel_root,
            engine_root=engine_root,
        )
        self.deterministic = deterministic
        self.decision_timeout_ms = decision_timeout_ms
        self.equity_source = equity_source
        self.equity_fallback_source = equity_fallback_source
        self.equity_iterations = equity_iterations
        self.observation_size = observation_size
        self.load_error = ""
        self.fallback_counts = {
            "no_model": 0,
            "inference_errors": 0,
            "illegal_actions": 0,
            "timeouts": 0,
        }
        self.equity_counts = {
            "computed": 0,
            "fallbacks": 0,
        }
        self.model = None
        try:
            self.model = load_reduced_checkpoint_model(
                self.checkpoint_path,
                basemodel_root=self.basemodel_root,
                observation_size=observation_size,
            )
        except Exception as exc:
            self.load_error = f"reduced checkpoint load failed: {exc}"
            if require_checkpoint:
                raise RuntimeError(self.load_error) from exc

    def _timed_out(self, started: float) -> bool:
        if self.decision_timeout_ms is None:
            return False
        return ((time.perf_counter() - started) * 1000.0) > float(self.decision_timeout_ms)

    def _state_with_equity(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        add_basemodel_to_path(self.basemodel_root)
        from poker_ai.equity import EquityConfig, compute_hero_equity

        enriched = dict(game_state)
        equity, source_used = compute_hero_equity(
            enriched,
            EquityConfig(
                source=self.equity_source,
                fallback_source=self.equity_fallback_source,
                iterations=self.equity_iterations,
            ),
        )
        enriched["hero_equity"] = equity
        enriched["hero_equity_source"] = source_used
        self.equity_counts["computed"] += 1
        if source_used != self.equity_source:
            self.equity_counts["fallbacks"] += 1
        return enriched

    def _select_reduced_action(self, game_state: Dict[str, Any]) -> tuple[int, list[bool], list[float]]:
        if self.model is None:
            raise RuntimeError("reduced model is not loaded")
        add_basemodel_to_path(self.basemodel_root)
        import torch

        from poker_ai.actions import legal_action_mask
        from poker_ai.reduced_observations import encode_reduced_observation

        observation = encode_reduced_observation(game_state)
        legal_mask = legal_action_mask(game_state)
        if not any(legal_mask):
            raise ValueError("game_state produced no legal actions")
        obs_tensor = torch.tensor([observation], dtype=torch.float32)
        mask_tensor = torch.tensor([legal_mask], dtype=torch.bool)
        with torch.no_grad():
            logits, _values = self.model(obs_tensor)
            logits = logits.masked_fill(~mask_tensor, -1.0e9)
            if self.deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
        return int(action.item()), legal_mask, observation

    def get_action(self, game_state: Dict[str, Any]) -> EngineAction:
        started = time.perf_counter()
        if has_forbidden_state_keys(game_state):
            return conservative_engine_action(game_state)
        if self.model is None:
            self.fallback_counts["no_model"] += 1
            return conservative_engine_action(game_state)
        try:
            add_basemodel_to_path(self.basemodel_root)
            from poker_ai.actions import action_to_engine

            model_state = self._state_with_equity(game_state)
            action_id, legal_mask, _observation = self._select_reduced_action(model_state)
            if self._timed_out(started):
                self.fallback_counts["timeouts"] += 1
                return conservative_engine_action(game_state)
            if action_id < 0 or action_id >= len(legal_mask) or not legal_mask[action_id]:
                self.fallback_counts["illegal_actions"] += 1
                return conservative_engine_action(game_state)
            action = action_to_engine(action_id, model_state)
            if not is_engine_action(action):
                self.fallback_counts["illegal_actions"] += 1
                return conservative_engine_action(game_state)
            return action
        except Exception:
            self.fallback_counts["inference_errors"] += 1
            return conservative_engine_action(game_state)
