"""No-network checks for supplemental Hermes replay selection and analysis."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hermes_monitor_rules import replay_sources  # noqa: E402
from supplement_hermes_monitors import _activate, _phases, _replay_config, _resolve_batches  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SupplementTests(unittest.TestCase):
    def test_legacy_config_migrates_in_attempt_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "run" / "hermes.container.yaml"
            original.parent.mkdir()
            original.write_text("pro2guard:\n  enabled: true\n  unsafe_states: []\n  horizon: 20\nharness: ../harness\n")
            attempt = root / "attempt"
            attempt.mkdir()
            config = {"pro2guard": {"enabled": True, "unsafe_states": [], "horizon": 20},
                      "harness": "../harness"}
            migrated, removed = _replay_config(original, attempt, config)
            self.assertEqual(removed, ["unsafe_states", "horizon"])
            self.assertEqual(migrated.parent, attempt)
            self.assertIn("unsafe_states: []", original.read_text())
            self.assertNotIn("unsafe_states", migrated.read_text())
            self.assertIn(str(root / "harness"), migrated.read_text())

    def test_asb_and_privacy_analyzers_use_supplement(self) -> None:
        for benchmark, analyzer, evaluation in (
            (
                "agent_security_bench", "analyze_hermes_asb_monitors.py",
                {"attack_success": True, "utility": True, "called_tools": []},
            ),
            (
                "privacylens_live", "analyze_hermes_privacylens_live_monitors.py",
                {"benchmark": "privacylens_live", "has_leakage": True, "utility": True},
            ),
        ):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as temp:
                batch = Path(temp) / "batch"
                phase = batch / "run_001" / "target"
                write_json(batch / "summary.json", {
                    "config": {"agent.yaml": {benchmark: {"enabled": True}}},
                    "items": [{"index": 1}], "runs": 1,
                })
                write_json(phase / "evaluation.json", evaluation)
                write_json(phase / "defenses.json", {
                    "mode": "replay", "methods": {"aegis": {"status": "failed", "error": "old failure"}},
                })
                attempt = phase / "defense_supplements" / "trial" / "aegis"
                replacement = attempt / "defense_replay" / "aegis" / "defenses.json"
                write_json(replacement, {"trace": [{"decisions": {
                    "_last_aegis_decision": {"cascade_decision": "block"},
                }}]})
                _activate(phase, "aegis", attempt, replacement, "test-image")
                subprocess.run([
                    sys.executable, str(ROOT / "scripts" / analyzer), str(batch),
                ], capture_output=True, text=True, check=True)
                metrics = json.loads((batch / "analysis" / "plugin_metrics.json").read_text())
                method = next(row for row in metrics["plugins"] if row["method"] == "aegis")
                if benchmark == "agent_security_bench":
                    self.assertEqual(method["target_attack_detection"]["matrix"], [[1, 0], [0, 0]])
                else:
                    self.assertEqual((method["tp"], method["errored_runs"]), (1, 0))
                self.assertIn(",True,", (batch / "analysis" / "run_labels.csv").read_text())

    def test_latest_paired_batch_and_failed_parent_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "all_monitors"
            for condition in ("skill_injection", "no_injection"):
                batch = root / condition / "20260101T000000Z_batch"
                write_json(batch / "summary.json", {"items": [{"index": 1, "ok": False}]})
                phase = batch / "run_001" / "target"
                write_json(phase / "defenses.json", {"mode": "replay", "methods": {}})
                write_json(batch / "run_001" / "result.json", {})
                (phase / "guard_lifecycle.jsonl").write_text("{}\n")
            write_json(root / "skill_injection" / "20260102T000000Z_unpaired" / "summary.json", {})
            batches = _resolve_batches(root, "")
            self.assertEqual({p.name for p in batches}, {"20260101T000000Z_batch"})
            self.assertEqual(len(_phases(batches[0], "agentdojo")), 1)

    def test_supplement_overlays_original_and_updates_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "gpt"
            original_bytes = {}
            for condition, verdict, attack_success in (
                ("skill_injection", "block", True),
                ("no_injection", "allow", False),
            ):
                batch = root / "all_monitors" / condition / "20260101T000000Z_batch"
                phase = batch / "run_001" / "target"
                write_json(batch / "summary.json", {"items": [{"index": 1}]})
                write_json(batch / "run_001" / "result.json", {
                    "harness": {"agentdojo": {"attack_success": attack_success, "utility": True}}
                })
                write_json(phase / "defenses.json", {
                    "mode": "replay", "methods": {"aegis": {"status": "failed", "error": "old failure"}}
                })
                old = phase / "defense_replay" / "aegis" / "defenses.json"
                write_json(old, {"trace": [{"decisions": {"_last_aegis_decision": {"error": "old failure"}}}]})
                original_bytes[condition] = old.read_bytes()
                attempt = phase / "defense_supplements" / "trial" / "aegis"
                replacement = attempt / "defense_replay" / "aegis" / "defenses.json"
                write_json(replacement, {"trace": [{"decisions": {
                    "_last_aegis_decision": {"cascade_decision": verdict, "allowed": verdict != "block"}
                }}]})
                _activate(phase, "aegis", attempt, replacement, "test-image")
                sources = replay_sources(phase, json.loads((phase / "defenses.json").read_text()))
                self.assertEqual(sources["aegis"]["path"], replacement)
                self.assertTrue(sources["aegis"]["supplemented"])
                self.assertEqual(old.read_bytes(), original_bytes[condition])
            subprocess.run([
                sys.executable, str(ROOT / "scripts/analyze_hermes_agentdojo_monitors.py"),
                "--root", str(root), "--batch", "20260101T000000Z_batch",
            ], capture_output=True, text=True, check=True)
            rows = json.loads((root / "analysis" / "confusion_matrices.json").read_text())
            self.assertEqual(rows[0]["injection_matrix"], [[1, 0], [0, 1]])
            self.assertEqual(rows[0]["errors_on"], 0)
            self.assertEqual(rows[0]["errors_off"], 0)
            self.assertIn("supplemented", (root / "analysis" / "run_labels.csv").read_text().splitlines()[0])

    def test_overlay_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            phase = Path(temp)
            write_json(phase / "defense_supplements.json", {
                "version": 1, "methods": {"aegis": {"path": "../outside.json"}}
            })
            with self.assertRaisesRegex(ValueError, "Unsafe supplement path"):
                replay_sources(phase, {"methods": {}})


if __name__ == "__main__":
    unittest.main()
