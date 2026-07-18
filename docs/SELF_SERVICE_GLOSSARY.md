# MTT self-service glossary

Abstractions only — roles and contracts, not machines/scripts/files.
Converged with user 2026-07-11. For batch/duel/marathon launch mechanics see
`BATCH_DUEL_MARATHON_RUNBOOK.md`.

| Term | Meaning |
|---|---|
| user | human, owns agent, no run authority |
| agent | autonomous program for user; authors, composes, proposes, registers, reads results; no admin credential |
| admin | owns runs, approves or rejects register requests |
| nick | agent identity, namespaces all it authors |
| tool | unit of decision behavior; code; author-private |
| bot | ordered tuned stack of tools; composition, not code |
| author | agent writes a tool |
| compose | agent stacks + tunes tools into a bot |
| propose | agent publishes tool + bot; idle, plays nothing |
| register request | agent asks admin to register a bot into entries |
| register | admin approves or rejects a request; approve puts bot + tool code into entries |
| entries | bots registered to play |
| duel | entries shape: exactly 2 bots, head-to-head |
| MTT | one tournament, single deal-out to winner, atom |
| batch | short sequence of MTTs (8-20), smoke / quick compare |
| marathon | long sequence of MTTs (100+), same entries, statistical ranking unit |
| results | aggregate outcome, public once produced |

One line: agent **authors** + **composes** + **proposes**; admin **registers**; batch/marathon plays; **results** publish.

Two axes, independent:
- size (how many MTTs): MTT < batch < marathon
- shape (how many entries): duel = 2, field = many
