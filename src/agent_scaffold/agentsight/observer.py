from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from ..config import AgentSightConfig


class AgentSightObserver:
    """Own one AgentSight capture process for a single scaffold job."""

    def __init__(
        self,
        config: AgentSightConfig,
        run_dir: Path,
        *,
        target_pid: int,
        binary_path: str = "",
    ) -> None:
        self.config = config
        self.run_dir = run_dir.resolve()
        self.target_pid = int(target_pid)
        self.binary_path = binary_path
        self.db_path = self._artifact_path(config.db_path)
        self.snapshot_path = self._artifact_path(config.snapshot_path)
        self.log_path = self._artifact_path(config.log_path)
        self.status_path = self.run_dir / "agentsight_status.json"
        self._process: subprocess.Popen[str] | None = None
        self._log: Any = None
        self._prefix: list[str] = []
        self._status: dict[str, Any] = {
            "enabled": True,
            "status": "initializing",
            "capture": config.capture,
            "target_pid": self.target_pid,
            "binary_path": binary_path,
            "db_path": str(self.db_path),
            "snapshot_path": str(self.snapshot_path),
            "log_path": str(self.log_path),
            "status_path": str(self.status_path),
            "started_at": None,
            "completed_at": None,
            "returncode": None,
            "error": "",
        }

    def _artifact_path(self, value: str) -> Path:
        path = (self.run_dir / value).resolve()
        try:
            path.relative_to(self.run_dir)
        except ValueError as exc:
            raise ValueError(f"AgentSight artifact escapes job directory: {value}") from exc
        return path

    def _write_status(self) -> None:
        self.status_path.write_text(
            json.dumps(self._status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _resolve_prefix(self) -> list[str]:
        binary = self.config.binary
        expanded = str(Path(binary).expanduser()) if os.sep in binary else binary
        if os.sep not in binary and shutil.which(binary) is None:
            raise RuntimeError(f"AgentSight executable not found: {binary}")
        if os.sep in binary and not Path(expanded).exists():
            raise RuntimeError(f"AgentSight executable not found: {binary}")
        self.config.binary = expanded
        if self.config.privilege == "none" or (hasattr(os, "geteuid") and os.geteuid() == 0):
            return []
        sudo = shutil.which("sudo")
        if sudo is not None:
            probe = subprocess.run(
                [sudo, "-n", "true"],
                text=True,
                capture_output=True,
                check=False,
            )
            if probe.returncode == 0:
                return [sudo, "-n"]
        if self.config.privilege == "sudo":
            raise RuntimeError(
                "AgentSight privilege=sudo requires non-interactive sudo authorization"
            )
        return []

    def _capture_command(self) -> list[str]:
        common = ["--db", str(self.db_path)]
        if self.config.capture == "full":
            command = [
                *self._prefix,
                self.config.binary,
                "record",
                "-p",
                str(self.target_pid),
            ]
            if self.binary_path:
                command.extend(["--binary-path", self.binary_path])
            command.extend(common)
            if self.config.web_server:
                command.extend(["--server-port", str(self.config.server_port)])
            else:
                command.append("--no-server")
            return command

        command = [
            *self._prefix,
            self.config.binary,
            "debug",
            "trace",
            "--ssl=false",
            "--process=true",
            "--system",
            "-p",
            str(self.target_pid),
            "--quiet",
            *common,
        ]
        if self.binary_path:
            command.extend(["--binary-path", self.binary_path])
        if self.config.web_server:
            command.extend(["--server", "--server-port", str(self.config.server_port)])
        return command

    def start(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.db_path, self.snapshot_path, self.log_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._prefix = self._resolve_prefix()
            command = self._capture_command()
            self._log = self.log_path.open("w", encoding="utf-8")
            self._log.write(f"command={json.dumps(command, ensure_ascii=False)}\n")
            self._log.flush()
            self._process = subprocess.Popen(
                command,
                cwd=str(self.run_dir),
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self._status.update(
                {"status": "starting", "started_at": time.time(), "command": command}
            )
            self._write_status()
            deadline = time.monotonic() + self.config.startup_timeout_seconds
            warmup_deadline = time.monotonic() + self.config.warmup_seconds
            while time.monotonic() < deadline:
                returncode = self._process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"AgentSight exited during startup with status {returncode}; "
                        f"see {self.log_path}"
                    )
                if time.monotonic() >= warmup_deadline:
                    self._status["status"] = "running"
                    self._write_status()
                    return self.result()
                time.sleep(0.05)
            raise RuntimeError(
                f"AgentSight did not become ready within {self.config.startup_timeout_seconds}s"
            )
        except Exception as exc:
            if self._process is not None and self._process.poll() is None:
                self._stop_process()
            self._status.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "completed_at": time.time(),
                    "returncode": self._process.poll() if self._process else None,
                }
            )
            self._write_status()
            self._close_log()
            if self.config.required:
                raise RuntimeError(f"Required AgentSight capture failed: {exc}") from exc
            return self.result()

    def _stop_process(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        try:
            os.killpg(self._process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            self._process.wait(timeout=self.config.shutdown_timeout_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self._process.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self._process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        self._process.wait(timeout=2)

    def _export_snapshot(self) -> None:
        if not self.db_path.exists():
            raise RuntimeError(f"AgentSight did not create its database: {self.db_path}")
        command = [
            *self._prefix,
            self.config.binary,
            "report",
            "--db",
            str(self.db_path),
            "export",
            "--output",
            str(self.snapshot_path),
        ]
        completed = subprocess.run(
            command,
            cwd=str(self.run_dir),
            text=True,
            capture_output=True,
            check=False,
            timeout=max(10.0, self.config.shutdown_timeout_seconds),
        )
        if self._log is not None:
            self._log.write("\nexport_command=" + json.dumps(command, ensure_ascii=False) + "\n")
            self._log.write(completed.stdout)
            self._log.write(completed.stderr)
            self._log.flush()
        if completed.returncode != 0:
            raise RuntimeError(
                f"AgentSight snapshot export failed with status {completed.returncode}; "
                f"see {self.log_path}"
            )

    def _restore_ownership(self) -> None:
        if len(self._prefix) != 2 or Path(self._prefix[0]).name != "sudo":
            return
        paths = [path for path in (self.db_path, self.snapshot_path) if path.exists()]
        if not paths:
            return
        command = ["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", *map(str, paths)]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0 and self._log is not None:
            self._log.write(f"ownership_warning={completed.stderr.strip()}\n")

    def stop(self) -> dict[str, Any]:
        if self._status["status"] == "failed" and self._process is None:
            return self.result()
        error = ""
        try:
            self._stop_process()
            if self._process is not None:
                self._status["returncode"] = self._process.poll()
            self._export_snapshot()
        except Exception as exc:
            error = str(exc)
        finally:
            self._restore_ownership()
            self._close_log()
        if error:
            self._status.update(
                {
                    "status": "failed" if self.config.required else "degraded",
                    "error": error,
                    "completed_at": time.time(),
                }
            )
            self._write_status()
            if self.config.required:
                raise RuntimeError(f"Required AgentSight finalization failed: {error}")
        else:
            self._status.update(
                {"status": "completed", "completed_at": time.time(), "error": ""}
            )
            self._write_status()
        return self.result()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def result(self) -> dict[str, Any]:
        return dict(self._status)
