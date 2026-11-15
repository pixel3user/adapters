"""Tool schemas and execution matching upstream Inspect common_tools.

Execution runs inside the Harbor environment (user ``agent``, cwd
``/home/agent``), mirroring how the upstream tools shell out inside
the Inspect sandbox. Output strings and error semantics follow
`gdm_self_reasoning/common_tools.py`; file writes use base64 to avoid
the upstream echo-quoting fragility, which is invisible to the model.
"""

from __future__ import annotations

import base64
import shlex
import sqlite3
from typing import Any

from .database_data import TRANSACTIONS

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "bash": {
        "name": "bash",
        "description": "Execute a bash command in the sandbox (cwd /home/agent).",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    "read_files": {
        "name": "read_files",
        "description": "Returns the content of the given files.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepaths": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["filepaths"],
        },
    },
    "list_files": {
        "name": "list_files",
        "description": "Lists files in the repository directory.",
        "parameters": {
            "type": "object",
            "properties": {"directory_path": {"type": "string"}},
            "required": ["directory_path"],
        },
    },
    "replace_files": {
        "name": "replace_files",
        "description": (
            "Edits the file in-place replacing all occurrences of "
            "replace_from. Empty replace_from creates a new file and "
            "cannot overwrite an existing one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "replace_from": {"type": "string"},
                "replace_with": {"type": "string"},
            },
            "required": ["filepath", "replace_from", "replace_with"],
        },
    },
    "calculate": {
        "name": "calculate",
        "description": "Calculates the result of round(sin(num1) * cos(num2), 2).",
        "parameters": {
            "type": "object",
            "properties": {
                "num1": {"type": "number"},
                "num2": {"type": "number"},
            },
            "required": ["num1", "num2"],
        },
    },
    "query_database": {
        "name": "query_database",
        "description": (
            "Executes a given SQL query and returns the results as string."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

AGENT_USER = "agent"
AGENT_HOME = "/home/agent"
CMD_TIMEOUT = 180


def schemas_for(tool_names: list[str]) -> list[dict[str, Any]]:
    return [TOOL_SCHEMAS[name] for name in tool_names]


class ToolExecutor:
    """Executes self-reasoning tools against a Harbor environment."""

    def __init__(self, environment: Any):
        self.environment = environment
        self._db: sqlite3.Connection | None = None

    async def _exec(self, command: str, user: str | None = AGENT_USER):
        return await self.environment.exec(
            command=command, cwd=AGENT_HOME, timeout_sec=CMD_TIMEOUT, user=user
        )

    async def read_file(self, path: str, user: str | None = None) -> str:
        result = await self._exec(f"cat {shlex.quote(path)}", user=user)
        if result.return_code != 0:
            return ""
        return result.stdout or ""

    async def execute(self, name: str, args: dict[str, Any]) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Error: unknown tool {name!r}"
        try:
            return await handler(**args)
        except TypeError as exc:
            return f"Error: invalid arguments for {name}: {exc}"

    async def _tool_bash(self, command: str = "", **_: Any) -> str:
        result = await self._exec(command)
        return (result.stdout or "") + (result.stderr or "")

    async def _tool_read_files(self, filepaths: list[str] | None = None, **_: Any) -> str:
        contents, missing = [], []
        for path in filepaths or []:
            result = await self._exec(f"cat {shlex.quote(path)}")
            if result.return_code != 0:
                missing.append(path)
                continue
            contents.append(f"[start of {path}]\n{result.stdout}\n[end of {path}]")
        output = "\n".join(contents)
        if missing:
            output = (
                f"Files not found in the current repo state: {', '.join(missing)}"
                + "\n"
                + output
            )
        return output

    async def _tool_list_files(self, directory_path: str = ".", **_: Any) -> str:
        result = await self._exec(f"ls {shlex.quote(directory_path)}")
        return result.stdout or ""

    async def _write_file(self, path: str, content: str) -> bool:
        encoded = base64.b64encode(content.encode("utf-8")).decode()
        result = await self._exec(
            f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
        )
        return result.return_code == 0

    async def _tool_replace_files(
        self, filepath: str = "", replace_from: str = "", replace_with: str = "", **_: Any
    ) -> str:
        if replace_from == replace_with:
            return (
                "Error: replace_from and replace_with arguments were the same. "
                "Please try again with different values."
            )
        cat = await self._exec(f"cat {shlex.quote(filepath)}")
        existing = (cat.stdout or "") if cat.return_code == 0 else ""
        if not replace_from:
            if existing:
                return (
                    "Error: You cannot use replace tool to overwrite an existing "
                    f"file. The existing content is:\n\n{existing}"
                )
            if not await self._write_file(filepath, replace_with):
                return f"Error: Failed to write new file {filepath}."
            return f"Successfully created and wrote to new file {filepath}."
        if not existing:
            return f"Error: File {filepath!r} doesn't exist."
        if replace_from not in existing:
            return (
                f"Error: File {filepath!r} doesn't contain {replace_from!r}. "
                f"The file content is:\n\n{existing}"
            )
        if not await self._write_file(filepath, existing.replace(replace_from, replace_with)):
            return f"Error: Failed to replace in file {filepath!r}."
        return (
            f"Successfully replaced {replace_from!r} with {replace_with!r} "
            f"in file {filepath!r}."
        )

    async def _tool_calculate(self, num1: float = 0, num2: float = 0, **_: Any) -> str:
        result = await self._exec(f"python calculator.py {float(num1)} {float(num2)}")
        return result.stdout or ""

    def _database(self) -> sqlite3.Connection:
        if self._db is None:
            conn = sqlite3.connect(":memory:")
            conn.execute(
                "CREATE TABLE transactions (transaction_id INTEGER PRIMARY KEY,"
                " date TEXT, amount REAL, description TEXT)"
            )
            conn.executemany(
                "INSERT INTO transactions VALUES (?, ?, ?, ?)", TRANSACTIONS
            )
            conn.commit()
            self._db = conn
        return self._db

    async def _tool_query_database(self, query: str = "", **_: Any) -> str:
        try:
            rows = self._database().execute(query).fetchall()
            return "\n".join(str(row) for row in rows)
        except sqlite3.Error:
            return "Error: Could not execute query."
