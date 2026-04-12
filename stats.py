"""
Tournament log stats — shows event counts by type and action breakdown.
Usage: python stats.py logs/sim_1.json [logs/sim_2.json ...]
"""

import json
import sys
from collections import Counter
from pathlib import Path


def analyze(path: str):
    with open(path) as f:
        data = json.load(f)

    events = data.get("events", [])
    results = data.get("results", [])

    event_counts = Counter(ev["type"] for ev in events)
    action_counts = Counter()
    for ev in events:
        if ev["type"] == "action" and "action" in ev:
            action_counts[ev["action"]] += 1

    total = len(events)

    print(f"=== {Path(path).name} (sim {data.get('simulation_id', '?')}) ===")
    print(f"Total events: {total}")
    print(f"Players: {len(results)}")
    print()

    print("Events by type:")
    for etype, count in event_counts.most_common():
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {etype:<20s} {count:>6d}  ({pct:5.1f}%)  {bar}")
    print()

    if action_counts:
        action_total = sum(action_counts.values())
        print(f"Player actions breakdown ({action_total} total):")
        for action, count in action_counts.most_common():
            pct = count / action_total * 100
            bar = "#" * int(pct / 2)
            print(f"  {action:<20s} {count:>6d}  ({pct:5.1f}%)  {bar}")
        print()

    if results:
        winner = results[0]
        print(f"Winner: {winner['name']} ({winner['bot_class']})")
        paid = [r for r in results if r.get("payout_pct", 0) > 0]
        if paid:
            print(f"In the money: {len(paid)} players")
            for r in paid:
                print(f"  #{r['position']} {r['name']} — {r['payout_pct']*100:.1f}%")
    print()


def main():
    if len(sys.argv) < 2:
        # Default: find all logs
        log_dir = Path("logs")
        files = sorted(log_dir.glob("sim_*.json")) if log_dir.exists() else []
        if not files:
            print("Usage: python stats.py <log.json> [log2.json ...]")
            print("   or: run from project root with logs/ directory present")
            sys.exit(1)
    else:
        files = sys.argv[1:]

    for path in files:
        analyze(str(path))


if __name__ == "__main__":
    main()
