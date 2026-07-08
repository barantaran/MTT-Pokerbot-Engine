# MTT marathon resume + durable event log

Tournament-level resume for `engine.fixed_bot_mtt_evaluation` marathons, plus
the durable per-hand event log it is built on. Added in commit `66ca377`.

The design target is cheap **SPOT** cloud instances that can be preempted and
destroyed at any time. A preempted marathon must be relaunchable without
throwing away the tournaments that already finished — and it must produce the
same aggregate report as an uninterrupted run.

## Model: resume at tournament granularity

A marathon is `mtt_count` independent tournaments, each run by a worker process
with its own seed. Resume operates on whole tournaments, not on hands:

| tournament state | on relaunch |
|---|---|
| **completed** — has a result checkpoint | loaded verbatim from disk, **skipped** |
| **interrupted / failed** — partial log, no checkpoint | stale partial log wiped, **re-run from scratch** |
| **not started** | run fresh |

There is **no mid-hand replay**. A re-run tournament is not resumed inside a
hand; it starts over. This is safe because each tournament is deterministic:

> `_seed_everything(seed)` seeds all RNG at worker start, so the same seed
> yields a **byte-identical** per-tournament result. Re-running a failed
> tournament from scratch reproduces exactly what a completed one would have
> produced — verified on real cloud hardware (all reruns byte-identical to the
> originals of an uninterrupted run).

Aggregation is order-independent, so loading completed tournaments in any order
and appending fresh reruns yields the same report as a single clean run.

## Config keys

Set in the marathon config JSON:

| key | default | meaning |
|---|---|---|
| `artifact_root` | `runs/<run_id>/<utc_stamp>` | where the log + checkpoints + report are written. **Set it to a stable path** (e.g. `runs/<run_id>`) so a relaunch resumes into the same dir. |
| `resume_log` | `true` | write per-hand chunk log + per-tournament completion checkpoints, and honor the resume skip-set. |
| `write_events` | `false` | also embed the full event list in the in-memory result / legacy `events/tournament_NNNN_events.json`. Independent of `resume_log`. |

Resume needs `resume_log: true` **and** a stable `artifact_root`. If
`artifact_root` defaults to a timestamped path, each launch is a fresh dir and
nothing is skipped.

## Artifact layout (under `artifact_root`)

```
<artifact_root>/
  events/mtt_<id>/chunk_00000.jsonl     # per-hand event chunks (write-once)
  events/mtt_<id>/chunk_00001.jsonl
  ...
  tournaments/mtt_<id>.result.json      # per-tournament completion checkpoint
  partial_summary.json                  # rewritten each heartbeat (local scratch)
  fixed_bot_evaluation_report.json      # final canonical report
```

- **chunk log** — the durable event stream, one tournament per subdir. Each
  finished hand becomes one new `chunk_NNNNN.jsonl` (one JSON event per line).
- **checkpoint** — `mtt_<id>.result.json`: the full tournament result minus the
  bulky `events` array (those live in the chunk log). **Its presence is what
  marks a tournament complete** for the skip-set. Nothing else drives skip — a
  partial chunk log without a checkpoint counts as interrupted.
- `partial_summary.json` is rewritten in place each heartbeat, so it stays on
  local scratch and is never a growing/rewritten object on a bucket mount.

## Why write-once chunk files

The log is a sequence of **write-once** chunk files, never a single appended /
rewritten object. This is what makes it correct on a **gcsfuse-mounted bucket**:

- gcsfuse uploads a file to GCS on `close()`. A new closed chunk per hand
  uploads only that chunk. An appended growing object would re-upload the whole
  log every time, and GCS objects are immutable anyway.
- A **background writer thread** (`ChunkLogWriter`) does the serialize + fsync +
  atomic rename + close, so upload latency stays off the marathon's hot path.
  The play loop only enqueues a batch; any writer error is re-raised from
  `close()`.
- The completion checkpoint is written **after** `writer.close()`, so all chunks
  of a tournament are guaranteed durable before it is marked done. A crash
  between the last chunk and the checkpoint just means the tournament re-runs —
  no corruption, no half-counted result.

## Components

- **`engine/event_log.py`**
  - `ChunkLogWriter(dir)` — bg-thread write-once chunk writer; `.write(events)`
    enqueues a hand's events, `.close()` drains + surfaces errors.
  - `atomic_write_json(path, payload)` — temp file + fsync + `os.replace`.
  - `completed_tournament_ids(root)` — scan `tournaments/` → `{id: checkpoint_path}`.
  - `load_tournament_result(path)`, `read_event_log(dir)`,
    `tournament_events_dir`, `tournament_result_path` — helpers.
- **`engine/tournament.py`** — `play(event_sink=None)`. When an `event_sink` is
  passed, events are flushed to it **per hand**; default `None` keeps the old
  batch-return behavior (backward compatible).
- **`engine/fixed_bot_mtt_evaluation.py`**
  - worker: builds the sink, wipes any stale partial log, plays, then writes the
    checkpoint (guarded by `resume_log`).
  - driver: builds the skip-set from `completed_tournament_ids`, ingests those
    results, and submits only the unfinished tournament ids. Logs:
    `resume: loaded N completed tournaments from <root>; running remaining M`.

## Running / resuming

Fresh run (writes log + checkpoints under a stable `artifact_root`):

```bash
PYTHONPATH=. python3 -m engine.fixed_bot_mtt_evaluation --config configs/<run>.json
# config has: "artifact_root": "runs/<run_id>", "resume_log": true
```

Resume after a crash / preemption — **re-run the exact same command**. The
driver finds the checkpoints, skips completed tournaments, and re-runs the rest.
No `--resume` flag; resume is implicit in pointing at a populated `artifact_root`.

In the cloud runner this is automatic: `job-runner.sh` sets
`artifact_root=<mount>/runs/<run_id>` (stable per `run_id`), so relaunching the
same `run_id` resumes off the gcsfuse-mounted bucket. See the `mtt-cloud-runner`
repo README.

## Tests

`tests/test_resume_log.py` (all local, no VM):

- chunk-log roundtrip — write events → `read_event_log` returns them in order.
- skip-set — `completed_tournament_ids` picks up only tournaments with a checkpoint.
- interrupted rerun — a re-run tournament is byte-identical to the original.

```bash
PYTHONPATH=. python3 -m pytest tests/test_resume_log.py -q
```

## Scope / non-goals

- **No mid-hand replay.** The event log carries every card and action (audit /
  streaming / visualizer), but resume does not reconstruct in-flight hands from
  it — failed tournaments re-run from scratch. Injection-based replay was
  designed (see the runner plan) but is **out of scope** for this build.
- Resume is per-tournament, not per-hand. A tournament interrupted at hand 900
  of 1000 redoes all 1000 hands. Deterministic, so correct; just not minimal
  work.
