from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_scaffold.agentsight import AgentSightObserver, wait_for_start_gate
from agent_scaffold.config import AgentSightConfig, load_config
from agent_scaffold import container_runtime


def _completed(args: list[str], returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


class AgentSightConfigTests(unittest.TestCase):
    def _config(self, root: Path, agentsight: str) -> Path:
        path = root / "agent.yaml"
        path.write_text(
            "llm:\n"
            "  provider: openai\n"
            "  model: test\n"
            "agent:\n"
            "  name: test\n"
            "  system_prompt: test\n"
            "container: false\n"
            f"agentsight:\n{agentsight}",
            encoding="utf-8",
        )
        return path

    def test_agentsight_boolean_enables_full_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._config(Path(tmp), "  enabled: true\n")
            cfg = load_config(path)
        self.assertTrue(cfg.agentsight.enabled)
        self.assertEqual(cfg.agentsight.capture, "full")

    def test_agentsight_rejects_artifacts_outside_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._config(
                Path(tmp),
                "  enabled: true\n  db_path: ../outside.db\n",
            )
            with self.assertRaisesRegex(ValueError, "must stay inside"):
                load_config(path)

    def test_agentsight_rejects_invalid_capture_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._config(Path(tmp), "  enabled: true\n  capture: unknown\n")
            with self.assertRaisesRegex(ValueError, "capture"):
                load_config(path)


class AgentSightGateTests(unittest.TestCase):
    def test_gate_preloads_ssl_and_returns_when_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = Path(tmp) / "start"
            gate.touch()
            with mock.patch.dict(
                os.environ,
                {
                    "AGENTSIGHT_START_FILE": str(gate),
                    "AGENTSIGHT_READY_FILE": str(Path(tmp) / "ready"),
                    "AGENTSIGHT_START_TIMEOUT": "1",
                },
            ):
                wait_for_start_gate()
            import ssl

            self.assertTrue(ssl.OPENSSL_VERSION.startswith("OpenSSL"))
            self.assertTrue((Path(tmp) / "ready").exists())


class AgentSightObserverTests(unittest.TestCase):
    def _fake_agentsight(self, root: Path) -> Path:
        path = root / "agentsight"
        path.write_text(
            "#!/bin/sh\n"
            "db=''\n"
            "output=''\n"
            "previous=''\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$previous\" = db ]; then db=$arg; previous=''; continue; fi\n"
            "  if [ \"$previous\" = output ]; then output=$arg; previous=''; continue; fi\n"
            "  if [ \"$arg\" = --db ]; then previous=db; continue; fi\n"
            "  if [ \"$arg\" = --output ]; then previous=output; continue; fi\n"
            "done\n"
            "if [ \"$1\" = report ]; then\n"
            "  printf '{\"exported\":true}\\n' > \"$output\"\n"
            "  exit 0\n"
            "fi\n"
            "touch \"$db\"\n"
            "trap 'exit 0' INT TERM\n"
            "while :; do sleep 0.1; done\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_full_capture_lifecycle_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = self._fake_agentsight(root)
            cfg = AgentSightConfig(
                enabled=True,
                binary=str(binary),
                capture="full",
                privilege="none",
                warmup_seconds=0.05,
                startup_timeout_seconds=1,
                shutdown_timeout_seconds=1,
            )
            observer = AgentSightObserver(cfg, root / "job", target_pid=os.getpid())
            started = observer.start()
            self.assertEqual(started["status"], "running")
            result = observer.stop()
            self.assertEqual(result["status"], "completed")
            self.assertTrue(Path(result["db_path"]).exists())
            self.assertEqual(
                json.loads(Path(result["snapshot_path"]).read_text(encoding="utf-8")),
                {"exported": True},
            )
            log = Path(result["log_path"]).read_text(encoding="utf-8")
            self.assertIn('"record"', log)
            self.assertIn('"-p"', log)

    def test_relative_binary_is_resolved_before_job_cwd_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = self._fake_agentsight(root)
            previous = Path.cwd()
            try:
                os.chdir(root)
                cfg = AgentSightConfig(
                    enabled=True,
                    binary="./agentsight",
                    capture="full",
                    privilege="none",
                    warmup_seconds=0.05,
                    startup_timeout_seconds=1,
                    shutdown_timeout_seconds=1,
                )
                observer = AgentSightObserver(cfg, root / "job", target_pid=os.getpid())
                self.assertEqual(observer.start()["status"], "running")
                self.assertTrue(Path(cfg.binary).is_absolute())
                self.assertEqual(observer.stop()["status"], "completed")
            finally:
                os.chdir(previous)

    def test_system_capture_disables_ssl(self) -> None:
        cfg = AgentSightConfig(enabled=True, capture="system", privilege="none")
        with tempfile.TemporaryDirectory() as tmp:
            observer = AgentSightObserver(cfg, Path(tmp), target_pid=123)
            observer._prefix = []
            command = observer._capture_command()
        self.assertEqual(command[1:3], ["debug", "trace"])
        self.assertIn("--ssl=false", command)
        self.assertIn("--system", command)

    def test_missing_binary_is_fail_open_or_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            optional = AgentSightObserver(
                AgentSightConfig(enabled=True, binary="definitely-missing-agentsight", privilege="none"),
                Path(tmp) / "optional",
                target_pid=1,
            )
            self.assertEqual(optional.start()["status"], "failed")
            required = AgentSightObserver(
                AgentSightConfig(
                    enabled=True,
                    binary="definitely-missing-agentsight",
                    privilege="none",
                    required=True,
                ),
                Path(tmp) / "required",
                target_pid=1,
            )
            with self.assertRaisesRegex(RuntimeError, "Required AgentSight"):
                required.start()
            self.assertEqual(required.stop()["status"], "failed")
            self.assertIn("executable not found", required.stop()["error"])


class ContainerAgentSightTests(unittest.TestCase):
    def test_container_is_gated_and_observed_from_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "jobs" / "run"
            run_dir.mkdir(parents=True)
            config_path = workspace / "agent.yaml"
            config_path.write_text("test: true\n", encoding="utf-8")
            cfg = SimpleNamespace(
                container=SimpleNamespace(
                    image="agent:test",
                    auto_build=False,
                    dockerfile="Dockerfile",
                    workdir="/workspace",
                    network="host",
                    remove=True,
                    build_args={},
                    env=[],
                ),
                agentsight=AgentSightConfig(enabled=True, privilege="none"),
            )
            commands: list[list[str]] = []

            def fake_run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                if command[:3] == ["docker", "image", "inspect"]:
                    return _completed(command)
                if command[:2] == ["docker", "run"]:
                    (run_dir / "_agentsight_ready").touch()
                    return _completed(command, stdout="container-id\n")
                if command[:3] == ["docker", "inspect", "--format"]:
                    return _completed(command, stdout="4242\n")
                if command[:2] == ["docker", "wait"]:
                    (run_dir / "_container_result.json").write_text(
                        json.dumps({
                            "messages": [],
                            "harness": {},
                            "_trace_persist": {"output_path": "/workspace/jobs/run/trace.json"},
                        }), encoding="utf-8"
                    )
                    (run_dir / "_container_result.json").chmod(0o444)
                    (run_dir / "trace.json").write_text(
                        json.dumps({"harness": {"agentsight": {"status": "managed_by_host"}}}),
                        encoding="utf-8",
                    )
                    return _completed(command, stdout="0\n")
                if command[:2] == ["docker", "logs"]:
                    return _completed(command, stdout="agent output\n")
                if command[:3] == ["docker", "rm", "-f"]:
                    return _completed(command)
                raise AssertionError(f"unexpected command: {command}")

            observer_calls: list[tuple[int, str]] = []

            class FakeObserver:
                def __init__(self, config: AgentSightConfig, path: Path, *, target_pid: int, binary_path: str = "") -> None:
                    observer_calls.append((target_pid, binary_path))

                def start(self) -> dict[str, object]:
                    return {"enabled": True, "status": "running"}

                def stop(self) -> dict[str, object]:
                    return {"enabled": True, "status": "completed"}

            with mock.patch.object(container_runtime, "_run_checked", side_effect=fake_run), mock.patch.object(
                container_runtime, "AgentSightObserver", FakeObserver
            ):
                result = container_runtime.run_once_in_container(
                    cfg,
                    str(config_path),
                    "hello",
                    None,
                    None,
                    workspace,
                    run_dir,
                )

            docker_run = next(command for command in commands if command[:2] == ["docker", "run"])
            self.assertIn("-d", docker_run)
            self.assertIn("--name", docker_run)
            self.assertIn("AGENTSIGHT_MANAGED=1", docker_run)
            self.assertTrue(any(str(item).startswith("AGENTSIGHT_READY_FILE=") for item in docker_run))
            self.assertFalse("--privileged" in docker_run)
            self.assertEqual(observer_calls[0][0], 4242)
            self.assertTrue(observer_calls[0][1].startswith("docker://agent-scaffold-"))
            self.assertTrue((run_dir / "_agentsight_start").exists())
            self.assertEqual(result["harness"]["agentsight"]["status"], "completed")
            trace = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
            self.assertEqual(trace["harness"]["agentsight"]["status"], "completed")
            self.assertTrue(any(command[:3] == ["docker", "rm", "-f"] for command in commands))


if __name__ == "__main__":
    unittest.main()
