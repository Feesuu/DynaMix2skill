"""
Bash command execution tool.
"""

import os
import re
import shutil
import subprocess
import sys

# Add parent src to path for react_agent imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from react_agent import tool


_SECRET_ENV_NAME = re.compile(
    r"(?:^|_)(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)(?:$|_)",
    flags=re.IGNORECASE,
)


def _safe_tool_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not _SECRET_ENV_NAME.search(name) and not name.startswith("DYNAMIX_")
    }


def _sandboxed_command(command: str, working_dir: str) -> tuple[list[str] | str, bool]:
    if os.environ.get("DYNAMIX_TOOL_BWRAP", "").casefold() not in {
        "1",
        "true",
        "yes",
    }:
        return command, True
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("DYNAMIX_TOOL_BWRAP requires bubblewrap")
    resolved_working_dir = os.path.realpath(working_dir)
    argv = [
        bwrap,
        "--unshare-pid",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--dir",
        resolved_working_dir,
        "--bind",
        resolved_working_dir,
        resolved_working_dir,
        "--chdir",
        resolved_working_dir,
    ]
    public_root = os.environ.get("DYNAMIX_PUBLIC_SKILL_ROOT", "").strip()
    if public_root:
        public_root = os.path.realpath(public_root)
        argv.extend(("--dir", public_root, "--ro-bind", public_root, public_root))
    for value in os.environ.get("DYNAMIX_TOOL_MASK_PATHS", "").split(os.pathsep):
        path = os.path.realpath(value.strip()) if value.strip() else ""
        if path and path != "/" and os.path.exists(path):
            argv.extend(("--tmpfs", path))
    argv.extend(("/bin/sh", "-c", command))
    return argv, False


def create_bash_tool(working_dir: str, timeout: int = 120):
    """
    Create a bash execution tool for running commands.

    Args:
        working_dir: Directory where commands will be executed
        timeout: Command timeout in seconds
    """

    @tool(name="bash")
    def bash(command: str) -> str:
        """
        Execute a bash command in the working directory.
        Use this to run Python scripts, install packages, navigate files,
        or perform any shell operations.

        Args:
            command: The bash command to execute
        """
        try:
            execution_command, use_shell = _sandboxed_command(command, working_dir)
            result = subprocess.run(
                execution_command,
                shell=use_shell,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_safe_tool_environment(),
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}" if output else result.stderr
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"
                if "SyntaxError" in output:
                    if "python -c" in command:
                        output += (
                            "\n[Recovery hint] This `python -c` command has invalid Python syntax. "
                            "Do not retry the same one-line command. Write a multi-line `solution.py` "
                            "with a heredoc, then run `python solution.py`."
                        )
                    elif "solution.py" in output:
                        output += (
                            "\n[Recovery hint] `solution.py` is not valid Python. Do not retry the same file. "
                            "Simplify the code, avoid fragile nested quote formula strings, and write final "
                            "computed values directly when that satisfies the target cells."
                        )
            return output.strip() if output.strip() else "[Command completed with no output]"
        except subprocess.TimeoutExpired:
            return f"[ERROR] Command timed out after {timeout} seconds"
        except Exception as e:
            return f"[ERROR] Failed to execute command: {e}"

    return bash
