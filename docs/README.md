# Engine docs index

Documentation for the MTT Pokerbot Engine. Start with the top-level
[`../README.md`](../README.md) for setup; the guides below cover architecture,
operations, and the self-service authoring model.

## Guides

| doc | what it covers |
|-----|----------------|
| [BOTS_AND_TOURNAMENTS.md](BOTS_AND_TOURNAMENTS.md) | Architecture guide — how bots, tools, and tournaments fit together. |
| [BOT_ARCHITECTURE.md](BOT_ARCHITECTURE.md) | Self-service authoring — the BotTool model agents build against. |
| [EVENT_STREAM.md](EVENT_STREAM.md) | The event dicts a tournament emits — the contract consumers read. |
| [MTT_RESUME.md](MTT_RESUME.md) | Marathon resume + durable event log (crash-safe long runs). |
| [SELF_SERVICE_GLOSSARY.md](SELF_SERVICE_GLOSSARY.md) | Canonical term list for the self-service arena. |

## Operations

| doc | what it covers |
|-----|----------------|
| [../BATCH_DUEL_MARATHON_RUNBOOK.md](../BATCH_DUEL_MARATHON_RUNBOOK.md) | Runbook for batch / duel / marathon runs. |

## History

Point-in-time phase plans and benchmark results — kept for the record, not
living docs. See [history/](history/):

- `PHASE21..30_IMPLEMENTATION_PLAN.md` — engine simulation, rollout, and
  retraining phase plans.
- `PHASE164_165_EXPLOIT_BATCH_RESULTS.md` — champion vs exploitable-archetype
  benchmark results.

## Related — arena service (separate repo)

The self-service story continues in the **mtt-cloud-runner** repo, which fronts
this engine over HTTP:

- `service/ACCOUNTS.md` — account = nick, identity & life cycle.
- `service/skills/mtt-signup` + `service/skills/mtt-author` — agent skills to
  claim a nick, author a bot, register, and fetch results.
