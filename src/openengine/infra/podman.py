"""Container and system tool runner infrastructure."""

import shutil
import subprocess
from typing import List, Optional


class PodmanRunner:
  """Wrapper for executing containerized engineering solvers via Podman."""

  def __init__(self, image: str = "openengine-solvers:latest"):
    self.image = image
    self._podman_bin = shutil.which("podman") or shutil.which("docker")

  @property
  def is_available(self) -> bool:
    """Check if podman or docker executable is present and responsive."""
    if not self._podman_bin:
      return False
    try:
      res = subprocess.run(
          [self._podman_bin, "info"],
          capture_output=True,
          timeout=5,
          check=False,
      )
      return res.returncode == 0
    except Exception:
      return False

  def run_command(
      self,
      command: str,
      workspace_dir: str,
      env_vars: Optional[dict] = None,
      timeout_sec: Optional[int] = None,
  ) -> subprocess.CompletedProcess:
    """Run an arbitrary shell command inside the solver container."""
    if not self.is_available:
      raise RuntimeError(
          "Podman/Docker runtime is not available on this system."
      )

    cmd: List[str] = [
        self._podman_bin,
        "run",
        "--rm",
        "-v",
        f"{workspace_dir}:/workspace:Z",
        "-w",
        "/workspace",
    ]

    if env_vars:
      for k, v in env_vars.items():
        cmd.extend(["-e", f"{k}={v}"])

    cmd.append(self.image)
    cmd.append(f"/bin/bash -c '{command}'")

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=True,
    )
