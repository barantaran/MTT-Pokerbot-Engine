"""Redacted per-player hand replay slices.

Turns a run's raw event log into one paged, already-redacted slice per
population (in the arena, per nick), so a service can serve a player their own
hands without ever holding anyone else's hole cards.

Redaction happens **here**, once, at write time, against a single known viewer:

  * the viewer's own hole cards — always
  * the board — always
  * an opponent's hole cards — only if that opponent reached showdown, which is
    public information at a real table
  * everything else — never written, so it cannot leak

That is the whole reason the export is per-viewer rather than per-hand. A
shared per-hand file would have to be either unredacted or redacted at read
time; the first is a god view and the second puts god data in the reader's
memory on every request.

Usage:

    python -m engine.replay_export \
      --artifact-root runs/<run_id> \
      --report        runs/<run_id>/fixed_bot_evaluation_report.json \
      --run-id        <run_id> \
      --out           runs/<run_id>/replay \
      --fanout        <bucket-mount>/replays \
      --engine-sha    <sha>

The event stream it reads is documented in docs/EVENT_STREAM.md. Logs written
before that schema landed are rejected, not guessed at.
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from engine.event_log import atomic_write_json, read_event_log, tournament_events_dir

SCHEMA = "mtt-replay/1"
DEFAULT_PAGE_SIZE = 50
DEFAULT_MAX_HANDS_PER_NICK = 2000
REDACTION = "hero-only: own hole cards, the board, and hands shown at showdown"

# Path segment for a population name. Populations are usually nicks
# (^[a-z0-9_-]{3,32}$) but a house lineup can name one "<nick>.<bot>", so
# anything outside this set is replaced rather than trusted into a path.
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9_.-]")

STREET_ORDER = ("preflop", "flop", "turn", "river", "runout")


class SchemaError(RuntimeError):
    """The event log predates the schema this exporter requires."""


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def load_events(artifact_root: Path, tournament_id: int) -> List[Dict[str, Any]]:
    """Read one tournament's events, chunk log first, JSON array as fallback."""
    chunk_dir = tournament_events_dir(artifact_root, tournament_id)
    if chunk_dir.is_dir():
        events = read_event_log(chunk_dir)
        if events:
            return events
    flat = artifact_root / "events" / f"tournament_{tournament_id:04d}_events.json"
    if flat.is_file():
        with flat.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return []


def check_schema(events: Sequence[Dict[str, Any]], source: str) -> None:
    """Refuse a log this exporter cannot read correctly.

    There is deliberately no compatibility shim. A pre-schema log has raw treys
    integers where cards should be strings and no way to order events inside a
    hand once tables interleave; guessing produces a replay that is subtly wrong
    rather than one that is obviously missing.
    """
    start = next((e for e in events if e.get("type") == "hand_start"), None)
    if start is None:
        raise SchemaError(f"{source}: no hand_start event found")
    missing = [key for key in ("seq", "button_seat", "level") if key not in start]
    if missing:
        raise SchemaError(
            f"{source}: hand_start is missing {missing} — this log predates the "
            "replay event schema (docs/EVENT_STREAM.md). Re-run the tournament."
        )


def split_hands(events: Iterable[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
    """Group a flat, table-interleaved stream into per-hand event lists.

    Keyed on (table_id, hand_id) rather than on hand_start/hand_end adjacency,
    because Tournament.play emits one hand per table in turn into a single list.
    """
    open_hands: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for event in events:
        if "hand_id" not in event:
            continue  # tournament-scope event, not part of any hand
        key = (event.get("table_id"), event.get("hand_id"))
        bucket = open_hands.setdefault(key, [])
        bucket.append(event)
        if event.get("type") == "hand_end":
            yield sorted(open_hands.pop(key), key=lambda e: e.get("seq", 0))
    for key in sorted(open_hands, key=lambda k: (str(k[0]), str(k[1]))):
        # A hand with no hand_end (a crashed or truncated log) is still worth
        # exporting; it just ends where the log does.
        yield sorted(open_hands[key], key=lambda e: e.get("seq", 0))


# --------------------------------------------------------------------------
# hand assembly (unredacted, in-process only)
# --------------------------------------------------------------------------


def build_hand(events: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Fold one hand's events into a replay record. Cards are NOT yet redacted."""
    start = next((e for e in events if e.get("type") == "hand_start"), None)
    if start is None:
        return None

    seats: List[Dict[str, Any]] = []
    by_name: Dict[str, Dict[str, Any]] = {}
    for row in start.get("players", []):
        seat = {
            "seat": row.get("seat"),
            "name": row.get("name"),
            "position": row.get("position"),
            "stack_start": row.get("stack"),
            "stack_end": row.get("stack"),
            "cards": None,
            "revealed": False,
            "rank": None,
            "rank_class": None,
            "invested": 0,
            "won": 0,
            "net": 0,
            "folded_on": None,
            "all_in_on": None,
            "busted": False,
        }
        seats.append(seat)
        by_name[row.get("name")] = seat

    hand = {
        "ref": hand_ref(start),
        "tournament_id": start.get("tournament_id"),
        "table_id": start.get("table_id"),
        "hand_id": start.get("hand_id"),
        "level": start.get("level"),
        "blinds": start.get("blinds"),
        "started_at": start.get("started_at"),
        "ended_at": None,
        "button_seat": start.get("button_seat"),
        "seats": seats,
        "blinds_posted": [],
        "streets": [],
        "board": {"flop": [], "turn": [], "river": [], "runout": [], "final": []},
        "showdown": [],
        "payouts": [],
        "pot_total": 0,
        "awarded_total": 0,
        # Non-empty when the source events do not describe a coherent hand. The
        # exporter stays faithful to what the engine emitted rather than
        # papering over it; a consumer can show the hand and say so.
        "anomalies": [],
    }

    street = _open_street(hand, "preflop", [])
    pot = 0

    for event in events:
        kind = event.get("type")

        if kind == "post_blind":
            seat = by_name.get(event.get("player"))
            amount = int(event.get("amount") or 0)
            pot += amount
            hand["blinds_posted"].append({
                "seat": event.get("seat"),
                "name": event.get("player"),
                "blind": event.get("blind"),
                "amount": amount,
            })
            if seat is not None:
                seat["invested"] += amount
                if seat["invested"] >= seat["stack_start"]:
                    seat["all_in_on"] = "preflop"

        elif kind == "deal":
            seat = by_name.get(event.get("player"))
            if seat is not None:
                seat["cards"] = list(event.get("cards") or [])

        elif kind == "action":
            seat = by_name.get(event.get("player"))
            amount = int(event.get("amount") or 0)
            if amount < 0:
                # engine/table.py: a betting round that early-returns skips the
                # per-street current_bet reset, so call_amount can come out
                # negative and hand chips back. Real, pre-existing, and it makes
                # the pot walk backwards — flag the hand rather than hide it.
                hand["anomalies"].append(
                    f"negative amount {amount} on {event.get('player')} "
                    f"{event.get('action')} (seq {event.get('seq')})"
                )
            before = pot
            pot += amount
            action = {
                "seq": event.get("seq"),
                "seat": seat["seat"] if seat else None,
                "name": event.get("player"),
                "position": event.get("position"),
                "action": event.get("action"),
                "amount": amount,
                "call_amount": event.get("call_amount"),
                "pot_before": before,
                "pot_after": pot,
                "stack_after": None,
                "all_in": False,
                "tool_events": _tool_events(event),
            }
            if seat is not None:
                seat["invested"] += amount
                action["stack_after"] = seat["stack_start"] - seat["invested"]
                action["all_in"] = action["stack_after"] <= 0
                if action["all_in"] and seat["all_in_on"] is None:
                    seat["all_in_on"] = street["street"]
                if event.get("action") == "fold":
                    seat["folded_on"] = street["street"]
            street["actions"].append(action)

        elif kind == "board":
            cards = list(event.get("cards") or [])
            name = event.get("street") or "flop"
            hand["board"].setdefault(name, [])
            hand["board"][name].extend(cards)
            hand["board"]["final"].extend(cards)
            if street["street"] == name:
                # runout deals one card per event; keep them in one street
                street["board"] = list(hand["board"]["final"])
            else:
                street = _open_street(hand, name, list(hand["board"]["final"]), pot)

        elif kind == "showdown":
            seat = by_name.get(event.get("player"))
            hand["showdown"].append({
                "seat": seat["seat"] if seat else None,
                "name": event.get("player"),
                "cards": list(event.get("cards") or []),
                "rank": event.get("rank"),
                "rank_class": event.get("rank_class"),
            })

        elif kind == "award_pot":
            seat = by_name.get(event.get("player"))
            amount = int(event.get("amount") or 0)
            hand["awarded_total"] += amount
            hand["payouts"].append({
                "seat": seat["seat"] if seat else None,
                "name": event.get("player"),
                "amount": amount,
                "showdown": bool(event.get("showdown")),
            })
            if seat is not None:
                seat["won"] += amount

        elif kind == "hand_end":
            hand["ended_at"] = event.get("ended_at")
            for row in event.get("players", []):
                seat = by_name.get(row.get("name"))
                if seat is not None:
                    seat["stack_end"] = row.get("stack")
                    seat["busted"] = row.get("stack") == 0

    street["pot_end"] = pot
    hand["pot_total"] = pot
    for seat in seats:
        seat["net"] = seat["won"] - seat["invested"]
    return hand


def _open_street(hand: Dict[str, Any], name: str, board: List[str], pot: int = 0) -> Dict[str, Any]:
    if hand["streets"]:
        hand["streets"][-1]["pot_end"] = pot
    street = {"street": name, "board": board, "pot_start": pot, "pot_end": pot, "actions": []}
    hand["streets"].append(street)
    return street


def _tool_events(event: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The BotTool decisions behind one action, newest schema first.

    tool_events (all tools that fired) superseded tool_event (the last one);
    older logs carry only the singular key.
    """
    events = event.get("tool_events")
    if isinstance(events, list) and events:
        return events
    single = event.get("tool_event")
    if isinstance(single, dict) and single:
        return [single]
    return None


def hand_ref(start: Dict[str, Any]) -> str:
    return "t{}.b{}.h{}".format(
        start.get("tournament_id"), start.get("table_id"), start.get("hand_id")
    )


# --------------------------------------------------------------------------
# redaction
# --------------------------------------------------------------------------


def redact(hand: Dict[str, Any], hero_names: Sequence[str]) -> Dict[str, Any]:
    """Project one assembled hand down to what `hero_names` may see.

    Everything an opponent held is dropped here, not hidden downstream: a key
    that is never written cannot be leaked by a bug in whatever serves it.
    """
    heroes = set(hero_names)
    revealed = {row["name"]: row for row in hand["showdown"]}

    seats = []
    for seat in hand["seats"]:
        name = seat["name"]
        is_hero = name in heroes
        shown = revealed.get(name)
        out = {
            "seat": seat["seat"],
            "name": name,
            "position": seat["position"],
            "stack_start": seat["stack_start"],
            "stack_end": seat["stack_end"],
            "is_hero": is_hero,
            "invested": seat["invested"],
            "won": seat["won"],
            "net": seat["net"],
            "folded_on": seat["folded_on"],
            "all_in_on": seat["all_in_on"],
            "busted": seat["busted"],
            "cards": None,
            "revealed": False,
            "rank": None,
            "rank_class": None,
        }
        if is_hero:
            out["cards"] = seat["cards"]
        if shown is not None:
            # A hand shown at showdown is public; keep the engine's own reveal.
            out["cards"] = shown["cards"]
            out["revealed"] = True
            out["rank"] = shown["rank"]
            out["rank_class"] = shown["rank_class"]
        out["hidden"] = out["cards"] is None
        seats.append(out)

    streets = []
    for street in hand["streets"]:
        actions = []
        for action in street["actions"]:
            row = dict(action)
            # tool_event payloads are author-controlled free-form dicts
            # (engine/bot_tools.py, attached in engine/table.py) — an author can
            # put their own hole cards in one. Hero keeps their own; nobody else's
            # survive.
            if row["name"] not in heroes:
                row["tool_events"] = None
            actions.append(row)
        streets.append({**street, "actions": actions})

    hero_seats = [seat["seat"] for seat in seats if seat["is_hero"]]
    return {
        "ref": hand["ref"],
        "tournament_id": hand["tournament_id"],
        "table_id": hand["table_id"],
        "hand_id": hand["hand_id"],
        "level": hand["level"],
        "blinds": hand["blinds"],
        "started_at": hand["started_at"],
        "ended_at": hand["ended_at"],
        "button_seat": hand["button_seat"],
        "hero_seats": hero_seats,
        "seats": seats,
        "blinds_posted": hand["blinds_posted"],
        "streets": streets,
        "board": hand["board"],
        "showdown": hand["showdown"],
        "payouts": hand["payouts"],
        "pot_total": hand["pot_total"],
        "awarded_total": hand["awarded_total"],
        "anomalies": hand["anomalies"],
        "redacted": True,
    }


def index_row(record: Dict[str, Any]) -> Dict[str, Any]:
    heroes = [seat for seat in record["seats"] if seat["is_hero"]]
    return {
        "ref": record["ref"],
        "tournament_id": record["tournament_id"],
        "table_id": record["table_id"],
        "hand_id": record["hand_id"],
        "level": record["level"],
        "blinds": record["blinds"],
        "hero_position": heroes[0]["position"] if heroes else None,
        "hero_stack_start": heroes[0]["stack_start"] if heroes else None,
        "hero_net": sum(seat["net"] for seat in heroes),
        "hero_showdown": any(seat["revealed"] for seat in heroes),
        "busted": any(seat["busted"] for seat in heroes),
        "pot_total": record["pot_total"],
        "showdown": bool(record["showdown"]),
        "players": len(record["seats"]),
    }


# --------------------------------------------------------------------------
# per-hero accumulation
# --------------------------------------------------------------------------


class HeroSlice:
    """Buffers one hero's hands, keeping at most `cap` of them.

    When the cap bites, the hands with the smallest absolute chip swing are
    dropped first: a marathon's 128,000 hands are mostly folded blinds, and the
    ones worth reviewing are the ones that moved a stack. Truncation is recorded
    in the index, never silent.
    """

    def __init__(self, name: str, cap: int) -> None:
        self.name = name
        self.slug = _SAFE_SEGMENT.sub("_", name)
        self.cap = cap
        self.seen = 0
        self.tournaments: set = set()
        self._heap: List[Tuple[int, int, Dict[str, Any]]] = []

    def add(self, record: Dict[str, Any]) -> None:
        self.seen += 1
        self.tournaments.add(record["tournament_id"])
        key = abs(index_row(record)["hero_net"] or 0)
        heapq.heappush(self._heap, (key, self.seen, record))
        if len(self._heap) > self.cap:
            heapq.heappop(self._heap)

    @property
    def truncated(self) -> bool:
        return self.seen > len(self._heap)

    def records(self) -> List[Dict[str, Any]]:
        return sorted(
            (record for _key, _order, record in self._heap),
            key=lambda r: (r["tournament_id"], r["table_id"], r["hand_id"]),
        )


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def write_slice(
    hero: HeroSlice,
    *,
    run_id: str,
    out_root: Path,
    page_size: int,
) -> Dict[str, Any]:
    records = hero.records()
    pages = [records[i:i + page_size] for i in range(0, len(records), page_size)] or [[]]
    hero_dir = out_root / hero.slug

    rows = []
    for page_number, page in enumerate(pages):
        for offset, record in enumerate(page):
            rows.append({**index_row(record), "page": page_number, "i": offset})
        atomic_write_json(
            hero_dir / f"hands_{page_number:03d}.json",
            {"schema": SCHEMA, "run_id": run_id, "nick": hero.name,
             "page": page_number, "hands": page},
        )

    index = {
        "schema": SCHEMA,
        "run_id": run_id,
        "nick": hero.name,
        "page_size": page_size,
        "pages": len(pages),
        "hands": len(records),
        "hands_total": hero.seen,
        "truncated": hero.truncated,
        "selection": "top_by_abs_net" if hero.truncated else "all",
        "tournaments": sorted(hero.tournaments),
        "rows": rows,
    }
    atomic_write_json(hero_dir / "index.json", index)
    return index


def export(
    *,
    artifact_root: Path,
    report: Dict[str, Any],
    run_id: str,
    out_root: Path,
    fanout_root: Optional[Path] = None,
    engine_sha: str = "unknown",
    nicks: Optional[Sequence[str]] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_hands_per_nick: int = DEFAULT_MAX_HANDS_PER_NICK,
) -> Dict[str, Any]:
    name_to_population = dict(report.get("name_to_population") or {})
    if not name_to_population:
        raise SchemaError(
            "report has no name_to_population — it predates the replay export "
            "(engine/fixed_bot_mtt_evaluation.py). Re-run the marathon."
        )

    wanted = set(nicks) if nicks else set(name_to_population.values())
    heroes: Dict[str, HeroSlice] = {
        population: HeroSlice(population, max_hands_per_nick)
        for population in sorted(wanted)
        if population in set(name_to_population.values())
    }
    if not heroes:
        raise SchemaError(f"no populations to export (asked for {sorted(wanted)})")

    names_by_population: Dict[str, List[str]] = {}
    for name, population in name_to_population.items():
        names_by_population.setdefault(population, []).append(name)

    mtt_count = int(report.get("mtt_count") or 0)
    for tournament_id in range(1, mtt_count + 1):
        events = load_events(artifact_root, tournament_id)
        if not events:
            continue
        check_schema(events, f"tournament {tournament_id}")
        for hand_events in split_hands(events):
            hand = build_hand(hand_events)
            if hand is None:
                continue
            dealt = {seat["name"] for seat in hand["seats"] if seat["cards"]}
            for population, hero in heroes.items():
                hero_names = [n for n in names_by_population.get(population, []) if n in dealt]
                if not hero_names:
                    continue  # "hands he played" == hands he was dealt into
                hero.add(redact(hand, hero_names))

    engine_block = dict(report.get("engine") or {})
    manifest_nicks = []
    for population in sorted(heroes):
        hero = heroes[population]
        index = write_slice(hero, run_id=run_id, out_root=out_root, page_size=page_size)
        manifest_nicks.append({
            "nick": hero.name,
            "file": hero.slug,
            "player_names": sorted(names_by_population.get(population, [])),
            "hands": index["hands"],
            "hands_total": index["hands_total"],
            "pages": index["pages"],
            "truncated": index["truncated"],
            "tournaments": index["tournaments"],
        })

    manifest = {
        "schema": SCHEMA,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine_sha": engine_sha,
        "finished_at": report.get("finished_at"),
        "mtt_count": mtt_count,
        "table_size": engine_block.get("max_players_per_table"),
        "starting_stack": engine_block.get("starting_stack"),
        "hands_per_level": engine_block.get("hands_per_level"),
        "blinds_schedule": engine_block.get("blinds_schedule"),
        "payouts": engine_block.get("payouts"),
        "page_size": page_size,
        "redaction": REDACTION,
        "nicks": manifest_nicks,
    }
    # The manifest lands last, so its presence means the whole slice tree did.
    atomic_write_json(out_root / "index.json", manifest)

    if fanout_root is not None:
        for row in manifest_nicks:
            atomic_write_json(
                fanout_root / row["nick"] / f"{run_id}.json",
                {
                    "run_id": run_id,
                    "finished_at": manifest["finished_at"],
                    "hands": row["hands"],
                    "pages": row["pages"],
                    "page_size": page_size,
                    "truncated": row["truncated"],
                    "tournaments": row["tournaments"],
                },
            )
    return manifest


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fanout", type=Path, default=None)
    parser.add_argument("--engine-sha", default="unknown")
    parser.add_argument("--nicks", default="", help="comma-separated; default every population")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-hands-per-nick", type=int, default=DEFAULT_MAX_HANDS_PER_NICK)
    args = parser.parse_args(argv)

    with args.report.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    try:
        manifest = export(
            artifact_root=args.artifact_root,
            report=report,
            run_id=args.run_id,
            out_root=args.out,
            fanout_root=args.fanout,
            engine_sha=args.engine_sha,
            nicks=[n for n in args.nicks.split(",") if n] or None,
            page_size=args.page_size,
            max_hands_per_nick=args.max_hands_per_nick,
        )
    except SchemaError as exc:
        print(f"replay: {exc}", file=sys.stderr)
        return 2

    hands = sum(row["hands"] for row in manifest["nicks"])
    pages = sum(row["pages"] for row in manifest["nicks"])
    truncated = [row["nick"] for row in manifest["nicks"] if row["truncated"]]
    print(
        f"replay: {len(manifest['nicks'])} nicks, {hands} hands, {pages} pages -> {args.out}",
        flush=True,
    )
    if truncated:
        print(f"replay: truncated to {args.max_hands_per_nick} hands for {truncated}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
