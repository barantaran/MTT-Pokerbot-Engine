"""Durable per-hand event logging for resumable MTT runs.

The log is written as a sequence of write-once chunk files
(``<dir>/chunk_NNNNN.jsonl``), one JSON event per line. Writing whole files
that are then closed — rather than appending to a single growing object — is
what lets a gcsfuse-mounted bucket upload each chunk on ``close()`` without
re-uploading the whole log, and it matches GCS object immutability. A
background writer thread performs the ``close()`` so upload latency stays off
the marathon's hot path.

A finished tournament also writes an atomic result checkpoint
(``tournaments/mtt_<id>.result.json``); its presence marks the tournament as
complete for the resume skip-set.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON to ``path`` atomically via a temp file + ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


class ChunkLogWriter:
    """Append-only event log written as write-once chunk files on a bg thread.

    Each ``write(events)`` enqueues one batch; the background thread serializes
    it to the next ``chunk_NNNNN.jsonl`` (temp file + fsync + atomic rename) and
    closes it. The caller only enqueues, so the close/upload latency never
    blocks play. Any writer error is re-raised from ``close()``.
    """

    def __init__(self, dir_path: str | Path) -> None:
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self._queue: "queue.Queue[Optional[Tuple[int, List[Dict[str, Any]]]]]" = queue.Queue()
        self._next_index = 0
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="chunk-log-writer", daemon=True)
        self._thread.start()

    def write(self, events: List[Dict[str, Any]]) -> None:
        if self._error is not None:
            raise self._error
        if not events:
            return
        index = self._next_index
        self._next_index += 1
        # Events are never mutated after creation, so a shallow list copy is
        # enough to hand the batch to the writer thread safely.
        self._queue.put((index, list(events)))

    __call__ = write

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                index, events = item
                self._write_chunk(index, events)
            except BaseException as exc:  # noqa: BLE001 - surfaced on close()
                if self._error is None:
                    self._error = exc
            finally:
                self._queue.task_done()

    def _write_chunk(self, index: int, events: List[Dict[str, Any]]) -> None:
        final = self.dir_path / f"chunk_{index:05d}.jsonl"
        tmp = final.with_name(f"{final.name}.tmp.{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, final)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error


def tournament_events_dir(artifact_root: str | Path, tournament_id: int) -> Path:
    return Path(artifact_root) / "events" / f"mtt_{tournament_id}"


def tournament_result_path(artifact_root: str | Path, tournament_id: int) -> Path:
    return Path(artifact_root) / "tournaments" / f"mtt_{tournament_id}.result.json"


def completed_tournament_ids(artifact_root: str | Path) -> Dict[int, Path]:
    """Map tournament_id -> result-checkpoint path for finished tournaments."""
    root = Path(artifact_root) / "tournaments"
    found: Dict[int, Path] = {}
    if not root.is_dir():
        return found
    suffix = ".result.json"
    for path in root.glob(f"mtt_*{suffix}"):
        stem = path.name[len("mtt_") : -len(suffix)]
        try:
            found[int(stem)] = path
        except ValueError:
            continue
    return found


def load_tournament_result(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_event_log(events_dir: str | Path) -> List[Dict[str, Any]]:
    """Concatenate all chunk files in order into a flat event list."""
    directory = Path(events_dir)
    events: List[Dict[str, Any]] = []
    if not directory.is_dir():
        return events
    for chunk in sorted(directory.glob("chunk_*.jsonl")):
        with chunk.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events
