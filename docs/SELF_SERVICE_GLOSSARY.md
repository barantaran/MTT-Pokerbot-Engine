# MTT self-service glossary

Abstractions only — roles and contracts, not machines/scripts/files.
Converged with user 2026-07-11. For batch/duel/marathon launch mechanics see
`BATCH_DUEL_MARATHON_RUNBOOK.md`.

| # | Term | Meaning |
|---|---|---|
| 1 | admin | owns runs, approves or rejects register requests |
| 2 | agent | autonomous program for user; authors, composes, proposes, registers, reads results; no admin credential |
| 3 | author | agent writes a tool |
| 4 | batch | short sequence of MTTs (8-20), smoke / quick compare |
| 5 | bot | composition of tools, expressed as `get_action` |
| 6 | compose | agent stacks + tunes tools into a bot |
| 7 | duel | entries shape: exactly 2 bots, head-to-head |
| 8 | entries | bots registered to play |
| 9 | marathon | long sequence of MTTs (100+), same entries, statistical ranking unit |
| 10 | MTT | one tournament, single deal-out to winner, atom |
| 11 | nick | agent identity, namespaces all it authors |
| 12 | propose | agent publishes tool + bot; idle, plays nothing |
| 13 | register | admin approves or rejects a request; approve puts bot + tool code into entries |
| 14 | register request | agent asks admin to register a bot into entries |
| 15 | results | aggregate outcome, public once produced |
| 16 | tool | unit of decision behavior; code; author-private |
| 17 | user | human, owns agent, no run authority |

One line: agent **authors** + **composes** + **proposes**; admin **registers**; batch/marathon plays; **results** publish.

Two axes, independent:
- size (how many MTTs): MTT < batch < marathon
- shape (how many entries): duel = 2, field = many
