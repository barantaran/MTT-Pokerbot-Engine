from __future__ import annotations

import io
import contextlib
import tempfile
import unittest
from pathlib import Path

from engine.event_log import (
    ChunkLogWriter,
    completed_tournament_ids,
    load_tournament_result,
    read_event_log,
    tournament_events_dir,
    tournament_result_path,
)
from engine.fixed_bot_mtt_evaluation import run_fixed_bot_evaluation


def _config(root: Path) -> dict:
    return {
        "run_id": "resume_log_test",
        "artifact_root": str(root),
        "mtt_count": 3,
        "workers": 2,
        "mtt_seed_start": 5000,
        "resume_log": True,
        "lineup": {"call": 6, "random": 6, "aggressive_no_equity": 6},
        "engine": {
            "equity_source": "constant",
            "equity_fallback_source": "constant",
            "starting_stack": 500,
            "max_players_per_table": 6,
            "hands_per_level": 3,
        },
    }


class ChunkLogWriterTests(unittest.TestCase):
    def test_writes_ordered_chunks_readable_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "events"
            w = ChunkLogWriter(d)
            w.write([{"type": "a", "i": 1}])
            w.write([{"type": "b", "i": 2}, {"type": "c", "i": 3}])
            w.write([])  # empty batch is a no-op
            w.close()
            chunks = sorted(d.glob("chunk_*.jsonl"))
            self.assertEqual([p.name for p in chunks], ["chunk_00000.jsonl", "chunk_00001.jsonl"])
            self.assertEqual([e["i"] for e in read_event_log(d)], [1, 2, 3])


class ResumeLogTests(unittest.TestCase):
    def test_log_checkpoint_and_skip_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            engine_root = Path(".")

            rep1 = run_fixed_bot_evaluation(_config(root), engine_root=engine_root)

            done = completed_tournament_ids(root)
            self.assertEqual(sorted(done), [1, 2, 3])
            for tid in (1, 2, 3):
                self.assertTrue(tournament_result_path(root, tid).exists())
                self.assertGreater(len(read_event_log(root / "events" / f"mtt_{tid}")), 0)

            # Re-run into the same artifact_root: all three are skipped (loaded
            # from checkpoints) and the report is identical.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rep2 = run_fixed_bot_evaluation(_config(root), engine_root=engine_root)
            self.assertIn("resume: loaded 3 completed tournaments", buf.getvalue())
            self.assertEqual(rep2["population_summary"], rep1["population_summary"])
            self.assertEqual(rep2["population_action_summary"], rep1["population_action_summary"])

    def test_the_durable_log_carries_the_replay_schema(self):
        """The chunk log is what the replay exporter reads, so assert its shape
        on the real written artifact, not just on Table's in-memory output."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            run_fixed_bot_evaluation(_config(root), engine_root=Path("."))

            events = read_event_log(tournament_events_dir(root, 1))
            self.assertGreater(len(events), 0)
            for event in events:
                if "hand_id" not in event:
                    continue  # tournament-scope event
                for key in ("table_id", "tournament_id", "seq"):
                    self.assertIn(key, event, event)
                for card in event.get("cards", []):
                    self.assertIsInstance(card, str, event)

    def test_interrupted_mtt_reruns_from_scratch(self):
        """A tournament with a partial log but no checkpoint is re-run fresh;
        completed tournaments are skipped. Re-run is deterministic (same seed),
        so its checkpoint is byte-identical to the original."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            engine_root = Path(".")

            run_fixed_bot_evaluation(_config(root), engine_root=engine_root)
            original_mtt2 = load_tournament_result(tournament_result_path(root, 2))

            # Simulate MTT #2 interrupted mid-run: drop its checkpoint, leave a
            # stale/garbage partial event log behind.
            tournament_result_path(root, 2).unlink()
            ev2 = tournament_events_dir(root, 2)
            import shutil

            shutil.rmtree(ev2)
            ev2.mkdir(parents=True)
            (ev2 / "chunk_00000.jsonl").write_text('{"type":"garbage"}\n', encoding="utf-8")

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_fixed_bot_evaluation(_config(root), engine_root=engine_root)
            self.assertIn("running remaining 1", buf.getvalue())

            self.assertEqual(sorted(completed_tournament_ids(root)), [1, 2, 3])
            rerun_mtt2 = load_tournament_result(tournament_result_path(root, 2))
            # Deterministic re-run -> identical result and clean (garbage wiped) log.
            self.assertEqual(rerun_mtt2["results"], original_mtt2["results"])
            self.assertEqual(rerun_mtt2["population_summary"], original_mtt2["population_summary"])
            log2 = read_event_log(tournament_events_dir(root, 2))
            self.assertFalse(any(e.get("type") == "garbage" for e in log2))
            self.assertGreater(len(log2), 0)


if __name__ == "__main__":
    unittest.main()
