from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from engine.bot_tools import available_decision_tools
from players.aggressive_bot import AggressiveBot
from players.aggressive_no_equity_bot import AggressiveNoEquityBot
from players.call_bot import CallBot
from players.ev_reaction_bot import EVFormulaBot, EVInitiativeBot, EVReactionBot
from players.icm_tight_bot import ICMTightBot
from players.noisy_equity_bot import NoisyEquityBot
from players.random_bot import RandomBot
from players.range_policy_bot import RangePolicyBot
from players.tight_equity_bot import TightEquityBot
from players.tournament_equity_bot import (
    AdaptiveTournamentICMEquityBot,
    ButtonStealTournamentICMEquityBot,
    ConfiguredTournamentEquityBot,
    TournamentEquityBot,
    TournamentEquityBotV2,
    TournamentICMEquityBot,
)


@dataclass(frozen=True)
class BotDefinition:
    bot_class: type
    population: str
    aliases: tuple[str, ...]
    default_params: Mapping[str, Any] | None = None


BOT_DEFINITIONS: tuple[BotDefinition, ...] = (
    BotDefinition(RandomBot, "random", ("random", "RandomBot")),
    BotDefinition(CallBot, "call", ("call", "calling_station", "CallBot")),
    BotDefinition(AggressiveNoEquityBot, "aggressive_no_equity", ("aggressive_no_equity", "AggressiveNoEquityBot")),
    BotDefinition(AggressiveBot, "aggressive_equity", ("aggressive_equity", "equity_aggressive", "aggressive", "AggressiveBot")),
    BotDefinition(TightEquityBot, "tight_equity", ("tight_equity", "tight", "TightEquityBot")),
    BotDefinition(NoisyEquityBot, "noisy_equity", ("noisy_equity", "noisy", "NoisyEquityBot")),
    BotDefinition(RangePolicyBot, "range_policy", ("range_policy", "RangePolicyBot")),
    BotDefinition(EVReactionBot, "ev_reaction", ("ev_reaction", "EVReactionBot")),
    BotDefinition(EVInitiativeBot, "ev_initiative", ("ev_initiative", "EVInitiativeBot")),
    BotDefinition(EVFormulaBot, "ev_formula", ("ev_formula", "EVFormulaBot")),
    BotDefinition(ICMTightBot, "icm_tight", ("icm_tight", "ICMTightBot")),
    BotDefinition(TournamentEquityBot, "tournament_equity", ("tournament_equity", "mtt_equity", "TournamentEquityBot")),
    BotDefinition(TournamentEquityBotV2, "tournament_equity_v2", ("tournament_equity_v2", "TournamentEquityBotV2")),
    BotDefinition(TournamentICMEquityBot, "tournament_icm_equity", ("tournament_icm_equity", "TournamentICMEquityBot")),
    BotDefinition(
        ConfiguredTournamentEquityBot,
        "configured_tournament_equity",
        ("configured_tournament_equity", "configured_mtt_equity", "ConfiguredTournamentEquityBot"),
    ),
    BotDefinition(
        AdaptiveTournamentICMEquityBot,
        "adaptive_tournament_icm_equity",
        ("adaptive_tournament_icm_equity", "AdaptiveTournamentICMEquityBot"),
    ),
    BotDefinition(
        ButtonStealTournamentICMEquityBot,
        "button_steal_tournament_icm_equity",
        ("button_steal_tournament_icm_equity", "ButtonStealTournamentICMEquityBot"),
    ),
)


BOT_REGISTRY: Dict[str, BotDefinition] = {
    alias: definition for definition in BOT_DEFINITIONS for alias in definition.aliases
}

_SPEC_KEYS = {
    "bot",
    "bot_type",
    "class",
    "count",
    "include_tool_set_in_name",
    "name",
    "name_prefix",
    "name_tool_set",
    "params",
    "population",
    "type",
}

_TOOL_NAME_ALIASES = {
    "icm_pressure": "icm",
    "preflop_reraise_tightness": "reraise",
    "table_adaptation": "table",
    "button_steal": "steal",
    "endgame_conversion": "endgame",
    "cbet_pressure": "cbet",
    "bluff_pressure": "bluff",
}

_POPULATION_NAME_ALIASES = {
    "configured_tournament_equity": "conf_mtt_eq",
    "configured_tournament_equity_baseline": "conf_mtt_eq",
    "configured_tournament_equity_range": "conf_mtt_eq_range",
    "configured_tournament_equity_legacy_range": "conf_mtt_eq_legacyrange",
    "configured_tournament_equity_adaptive_range": "conf_mtt_eq_adaptiverange",
    "configured_tournament_equity_player_range": "conf_mtt_eq_playerrange",
    "configured_tournament_equity_player_range_samplecurve": "conf_mtt_eq_playerrange_samplecurve",
    "configured_tournament_equity_player_range_nosampling": "conf_mtt_eq_playerrange_nosampling",
    "configured_tournament_equity_player_range_dampened": "conf_mtt_eq_playerrange_damped",
    "configured_tournament_equity_player_range_dampened_postflop_pot": "conf_mtt_eq_playerrange_damped_postfloppot",
    "configured_tournament_equity_player_range_dampened_postflop_pot_capped": "conf_mtt_eq_playerrange_damped_postfloppotcapped",
    "configured_tournament_equity_player_range_dampened_postflop_pot_capped_bluff": "conf_mtt_eq_playerrange_damped_postfloppotcapped",
    "configured_tournament_equity_player_range_dampened_postflop_pot_capped_bluffstrong": "conf_mtt_eq_playerrange_damped_postfloppotcapped_bluffstrong",
    "configured_tournament_equity_player_range_dampened_postflop_pot_capped_bluffleverage": "conf_mtt_eq_playerrange_damped_postfloppotcapped_bluffleverage",
}


def available_bot_tools() -> Dict[str, Dict[str, Any]]:
    tools: Dict[str, Dict[str, Any]] = {}
    seen: set[type] = set()
    for definition in BOT_DEFINITIONS:
        if definition.bot_class in seen:
            continue
        seen.add(definition.bot_class)
        tools[definition.population] = {
            "class": definition.bot_class.__name__,
            "aliases": list(definition.aliases),
            "params": _constructor_params(definition.bot_class),
        }
    for name, tool in available_decision_tools().items():
        tools[f"tool:{name}"] = tool
    return tools


def build_configurable_bots(
    lineup: Mapping[str, Any],
    engine_config: Mapping[str, Any],
    *,
    extra_specs: Iterable[Mapping[str, Any]] | None = None,
    legacy_population_map: Mapping[str, str] | None = None,
) -> tuple[List[Any], Dict[str, str]]:
    bots: List[Any] = []
    name_to_population: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    use_ranges = bool(engine_config.get("fixed_bots_use_preflop_spot_range", False))
    legacy_population_map = dict(legacy_population_map or {})

    def add_bot(bot: Any, population: str, name: str | None = None, name_prefix: str | None = None) -> None:
        if name is None:
            index = int(seen.get(population, 0)) + 1
            seen[population] = index
            prefix = name_prefix or population
            name = f"{prefix}_{index:03d}"
        bot.name = name
        bots.append(bot)
        name_to_population[bot.name] = population

    for key, count_value in lineup.items():
        count = int(count_value or 0)
        if count <= 0:
            continue
        bot_type = legacy_population_map.get(str(key), str(key))
        definition = _definition_for(bot_type)
        params = _default_params(definition, use_ranges=use_ranges)
        for _ in range(count):
            add_bot(_instantiate(definition, params), definition.population)

    for raw_spec in extra_specs or []:
        spec = dict(raw_spec)
        count = int(spec.get("count", 1) or 0)
        if count <= 0:
            continue
        bot_type = str(spec.get("type") or spec.get("bot") or spec.get("bot_type") or spec.get("class") or "")
        definition = _definition_for(bot_type)
        params = _default_params(definition, use_ranges=use_ranges)
        params.update(_inline_params(spec))
        params.update(dict(spec.get("params", {}) or {}))
        _validate_params(definition.bot_class, params)
        population = str(spec.get("population") or definition.population)
        name_prefix = str(spec.get("name_prefix") or "") or None
        if _include_tool_set_in_name(spec, definition, params):
            population = _population_with_tool_set(population, params)
            if name_prefix is not None:
                name_prefix = _population_with_tool_set(name_prefix, params)
        for index in range(count):
            name = str(spec.get("name") or "") or None
            if name is not None and count > 1:
                name = f"{name}_{index + 1:03d}"
            add_bot(
                _instantiate(definition, params),
                population,
                name=name,
                name_prefix=name_prefix,
            )

    return bots, name_to_population


def population_for_spec(spec: Mapping[str, Any], *, use_ranges: bool = False) -> str:
    raw_spec = dict(spec)
    bot_type = str(raw_spec.get("type") or raw_spec.get("bot") or raw_spec.get("bot_type") or raw_spec.get("class") or "")
    definition = _definition_for(bot_type)
    params = _default_params(definition, use_ranges=use_ranges)
    params.update(_inline_params(raw_spec))
    params.update(dict(raw_spec.get("params", {}) or {}))
    _validate_params(definition.bot_class, params)
    population = str(raw_spec.get("population") or definition.population)
    if _include_tool_set_in_name(raw_spec, definition, params):
        population = _population_with_tool_set(population, params)
    return population


def _definition_for(bot_type: str) -> BotDefinition:
    if not bot_type:
        raise ValueError("bot spec requires a type")
    try:
        return BOT_REGISTRY[bot_type]
    except KeyError as exc:
        available = ", ".join(sorted(BOT_REGISTRY))
        raise ValueError(f"unknown bot type {bot_type!r}; available bot types: {available}") from exc


def _constructor_params(bot_class: type) -> List[str]:
    signature = inspect.signature(bot_class.__init__)
    return [
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    ]


def _accepts_param(bot_class: type, key: str) -> bool:
    return key in _constructor_params(bot_class)


def _default_params(definition: BotDefinition, *, use_ranges: bool) -> Dict[str, Any]:
    params = dict(definition.default_params or {})
    if _accepts_param(definition.bot_class, "use_preflop_spot_range"):
        params.setdefault("use_preflop_spot_range", use_ranges)
    return params


def _inline_params(spec: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in spec.items()
        if key not in _SPEC_KEYS and value is not None
    }


def _validate_params(bot_class: type, params: Mapping[str, Any]) -> None:
    signature = inspect.signature(bot_class.__init__)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        return
    allowed = set(_constructor_params(bot_class))
    unknown = sorted(key for key in params if key not in allowed)
    if unknown:
        raise ValueError(
            f"{bot_class.__name__} does not accept params {unknown}; accepted params: {sorted(allowed)}"
        )


def _instantiate(definition: BotDefinition, params: Mapping[str, Any]) -> Any:
    _validate_params(definition.bot_class, params)
    return definition.bot_class(**dict(params))


def _include_tool_set_in_name(spec: Mapping[str, Any], definition: BotDefinition, params: Mapping[str, Any]) -> bool:
    explicit = spec.get("include_tool_set_in_name", spec.get("name_tool_set"))
    if explicit is not None:
        return bool(explicit)
    if spec.get("name") or spec.get("population"):
        return False
    return _accepts_param(definition.bot_class, "tools") and "tools" in params


def _population_with_tool_set(population: str, params: Mapping[str, Any]) -> str:
    return f"{_population_name_alias(population)}_{_tool_set_signature(params.get('tools'))}"


def _population_name_alias(population: str) -> str:
    return _POPULATION_NAME_ALIASES.get(population, population)


def _tool_set_signature(tools: Any) -> str:
    if tools is None:
        return "none"
    names: list[str] = []
    for raw_tool in tools:
        if isinstance(raw_tool, str):
            name = raw_tool
        elif isinstance(raw_tool, Mapping):
            name = str(raw_tool.get("type") or raw_tool.get("name") or "unknown")
        else:
            name = str(getattr(raw_tool, "name", "") or raw_tool.__class__.__name__)
        names.append(_slug(_TOOL_NAME_ALIASES.get(name, name)))
    return "_".join(name for name in names if name) or "none"


def _slug(value: str) -> str:
    chars = []
    previous_underscore = False
    for char in str(value).lower():
        if char.isalnum():
            chars.append(char)
            previous_underscore = False
        elif not previous_underscore:
            chars.append("_")
            previous_underscore = True
    return "".join(chars).strip("_")
