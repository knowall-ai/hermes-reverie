"""A small, dependency-free MCP stdio client.

Reverie's graph lives behind the `mcp-reverie` MCP server (binary ``reverie``), not behind a
Neo4j driver. This module is the whole transport: spawn the server as a child process, speak
JSON-RPC 2.0 over its stdin/stdout, do the MCP handshake (``initialize`` →
``notifications/initialized``), then ``tools/list`` and ``tools/call``.

Design notes:

* **One process, many turns.** Hermes may call from several turns at once, so every request
  is serialised under a lock and a reader thread demultiplexes responses by id. Server
  notifications and unmatched ids are dropped after being logged at debug level.
* **Restart on crash.** If the child has exited (or the pipe breaks mid-request) the next call
  respawns it and retries once. A server that dies during startup is reported with whatever it
  wrote to stderr, which is where MCP servers log.
* **No credentials in the log.** Values of environment variables that look like secrets are
  never logged; only their names are.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "hermes-reverie"
CLIENT_VERSION = "0.2.0"

#: Environment variables the server understands; everything else is inherited unchanged.
SERVER_ENV_PREFIXES = ("NEO4J_", "REVERIE_")
#: mcp-reverie tools that change nothing, so re-sending one after a failure is harmless.
READ_ONLY_TOOLS = frozenset({
    "search_memories", "query_memories", "memory_stats", "list_memory_labels", "get_guidance",
})
_SECRET_HINTS = ("PASSWORD", "SECRET", "TOKEN", "KEY", "AUTH")


def _is_secret(name: str) -> bool:
    upper = name.upper()
    return any(hint in upper for hint in _SECRET_HINTS)


def redact_env(env: Dict[str, str]) -> Dict[str, str]:
    """The same mapping with secret-looking values replaced — safe to log."""
    return {k: ("***" if _is_secret(k) else v) for k, v in env.items()}


class MCPError(RuntimeError):
    """Transport or protocol failure: the server would not start, answer, or spoke nonsense.

    Two flags decide whether the call may be sent again:

    ``delivered``
        False when the request provably never left the client (the process was gone before the
        write, or the write to its stdin failed). Retrying it cannot repeat anything.
    ``server_gone``
        True when the child process has exited. A timeout while the process is still **alive**
        leaves this False: the server is simply slow — the first embedding run downloads a model
        and can take minutes — and it is very likely still applying the call. Such a call is
        never retried, because that is how one ``create_memory`` becomes two nodes.
    """

    def __init__(self, message: str, delivered: bool = True, server_gone: bool = False) -> None:
        super().__init__(message)
        self.delivered = delivered
        self.server_gone = server_gone


class MCPToolError(RuntimeError):
    """The server answered, but the tool call itself failed (``isError``)."""


class MCPClient:
    """A JSON-RPC 2.0 client for one stdio MCP server process.

    Args:
        command: the server command, as a list or a shell-style string (default ``reverie``).
        env: extra environment for the child, layered over the parent's (secrets stay here).
        timeout: seconds to wait for a ``tools/call`` response.
        startup_timeout: seconds to wait for the ``initialize`` handshake.
        cwd: working directory for the child process.
    """

    def __init__(
        self,
        command: Sequence[str] | str = "reverie",
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        startup_timeout: float = 30.0,
        cwd: Optional[str] = None,
    ) -> None:
        self.command: List[str] = shlex.split(command) if isinstance(command, str) else list(command)
        if not self.command:
            raise ValueError("MCP server command is empty")
        self._extra_env = dict(env or {})
        self.timeout = float(timeout)
        self.startup_timeout = float(startup_timeout)
        self._cwd = cwd

        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._responses: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._stderr_reader: Optional[threading.Thread] = None
        self._stderr_tail: List[str] = []
        self._next_id = 0
        self._server_info: Dict[str, Any] = {}

    # -- process lifecycle -------------------------------------------------
    @property
    def server_info(self) -> Dict[str, Any]:
        return dict(self._server_info)

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def resolved_command(self) -> Optional[str]:
        """Absolute path of the server binary, or None when it is not on PATH."""
        return shutil.which(self.command[0])

    def _child_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(self._extra_env)
        return env

    def start(self) -> None:
        """Spawn the server and complete the MCP handshake. Idempotent."""
        with self._lock:
            if self.is_running():
                return
            self._shutdown_process()
            env = self._child_env()
            passed = sorted(k for k in self._extra_env if k.startswith(SERVER_ENV_PREFIXES))
            logger.debug("Reverie: starting MCP server %s (env: %s)", " ".join(self.command), ", ".join(passed))
            try:
                self._proc = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=self._cwd,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as exc:
                raise MCPError(f"MCP server not found: {self.command[0]} ({exc})") from exc
            except OSError as exc:
                raise MCPError(f"could not start MCP server {self.command[0]}: {exc}") from exc

            self._responses = queue.Queue()
            self._stderr_tail = []
            # The queue is handed to the reader so a thread left over from a dead process can
            # never push an EOF sentinel into the new session's queue.
            self._reader = threading.Thread(target=self._read_stdout, args=(self._proc, self._responses), daemon=True)
            self._reader.start()
            self._stderr_reader = threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True)
            self._stderr_reader.start()

            try:
                result = self._request(
                    "initialize",
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
                    },
                    timeout=self.startup_timeout,
                )
                self._server_info = (result or {}).get("serverInfo", {}) or {}
                self._notify("notifications/initialized", {})
            except Exception:
                self._shutdown_process()
                raise

    def stop(self) -> None:
        with self._lock:
            self._shutdown_process()

    close = stop

    def _shutdown_process(self) -> None:
        """Close stdin, end the process, then let the reader threads drain to EOF.

        Order matters: closing a pipe a reader thread is blocked on deadlocks on the buffer
        lock, so the child is stopped first and the readers are joined before anything else is
        closed.
        """
        proc, self._proc = self._proc, None
        reader, self._reader = self._reader, None
        stderr_reader, self._stderr_reader = self._stderr_reader, None
        self._server_info = {}
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()  # EOF: a well-behaved MCP server exits on its own
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
        for thread in (reader, stderr_reader):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    # -- pipes -------------------------------------------------------------
    def _read_stdout(self, proc: subprocess.Popen, responses: "queue.Queue[Dict[str, Any]]") -> None:
        stream = proc.stdout
        if stream is None:
            return
        try:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    logger.debug("Reverie: non-JSON line from MCP server: %s", line[:200])
                    continue
                if isinstance(message, dict) and "id" in message:
                    responses.put(message)
                else:
                    logger.debug("Reverie: MCP notification %s", (message or {}).get("method"))
        except (ValueError, OSError):
            pass  # pipe closed under us; the next call restarts the server
        finally:
            responses.put({"id": None, "_eof": True})

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        stream = proc.stderr
        if stream is None:
            return
        try:
            for line in stream:
                line = line.rstrip()
                if not line:
                    continue
                self._stderr_tail = (self._stderr_tail + [line])[-20:]
                logger.debug("Reverie MCP server: %s", line)
        except (ValueError, OSError):
            pass

    def _stderr_hint(self) -> str:
        return ("; server said: " + " | ".join(self._stderr_tail[-3:])) if self._stderr_tail else ""

    # -- JSON-RPC ----------------------------------------------------------
    def _send(self, payload: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise MCPError("MCP server is not running", delivered=False)
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            # The write failed, so the server never saw this request: safe to retry anything.
            raise MCPError(f"MCP server pipe closed: {exc}{self._stderr_hint()}", delivered=False) from exc

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: Dict[str, Any], timeout: Optional[float] = None) -> Any:
        """One request/response round trip. The caller holds the lock."""
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        # One deadline for the whole wait, not one per message: after a timeout the late answer
        # is still queued, and skipping it must not buy the next call another full timeout.
        deadline_timeout = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + deadline_timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise queue.Empty
                message = self._responses.get(timeout=remaining)
            except queue.Empty:
                # Slow, or dead without an EOF yet? Only the process itself can say.
                alive = self.is_running()
                raise MCPError(
                    f"MCP server did not answer {method} within {deadline_timeout:g}s"
                    + (" (the server is still running; it may still be applying the call)" if alive else "")
                    + self._stderr_hint(),
                    server_gone=not alive,
                )
            if message.get("_eof"):
                raise MCPError(f"MCP server exited during {method}{self._stderr_hint()}", server_gone=True)
            if message.get("id") != request_id:
                logger.debug("Reverie: ignoring stale MCP response id=%s", message.get("id"))
                continue
            if "error" in message:
                error = message["error"] or {}
                raise MCPError(f"{method} failed: {error.get('message', error)}")
            return message.get("result")

    # -- MCP surface -------------------------------------------------------
    def list_tools(self) -> List[Dict[str, Any]]:
        with self._lock:
            result = self._call_with_restart("tools/list", {}, repeatable=True)
        return list((result or {}).get("tools", []))

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None,
                  repeatable: Optional[bool] = None) -> Any:
        """Call a tool and return its parsed result.

        The MCP result is a list of content blocks; Reverie's server returns exactly one text
        block holding JSON, so the JSON is parsed and returned. Non-JSON text is returned as a
        string. ``isError`` results raise :class:`MCPToolError`.

        ``repeatable`` says whether sending this call twice is harmless. It defaults to True for
        the read-only tools and False for everything else, so a timed-out ``create_memory`` is
        never silently sent again — that is how one ``remember`` becomes two nodes.
        """
        payload = {"name": name, "arguments": dict(arguments or {})}
        if repeatable is None:
            repeatable = name in READ_ONLY_TOOLS
        with self._lock:
            result = self._call_with_restart("tools/call", payload, repeatable=repeatable)

        text = _content_text(result)
        if result.get("isError"):
            raise MCPToolError(text or f"{name} failed")
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return text

    def ping(self) -> bool:
        """True when the server is up and answering ``tools/list``."""
        try:
            return bool(self.list_tools())
        except Exception as exc:
            logger.debug("Reverie: MCP ping failed: %s", exc)
            return False

    # -- helpers -----------------------------------------------------------
    def _ensure_started(self) -> None:
        if not self.is_running():
            self.start()

    def _call_with_restart(self, method: str, params: Dict[str, Any],
                           repeatable: bool = False) -> Dict[str, Any]:
        """Run a request, restarting the server and retrying once — but only when that is safe.

        Three cases, in order of certainty:

        1. The request never left the client (``delivered=False``): retry anything.
        2. The process has exited (``server_gone``): retry only a repeatable (read-only) call.
           A write may have committed before the server died, so it is reported instead.
        3. The process is alive and merely slow: **never** retry, whatever the tool. A cold local
           embedding model can push the first ``create_memory`` past the timeout while the server
           goes on to apply it; a retry would create the node twice. The server is left running —
           its late answer is discarded as a stale id — and the caller gets the error.
        """
        self._ensure_started()
        try:
            return self._request(method, params) or {}
        except MCPError as exc:
            delivered = getattr(exc, "delivered", True)
            server_gone = getattr(exc, "server_gone", False)
            if delivered is not False:
                if not server_gone:
                    logger.warning("Reverie: %s timed out while the server is still running; not "
                                   "retrying, it may still be applying the call", method)
                    raise
                if not repeatable:
                    logger.warning("Reverie: the server died during %s; not retrying, the call may "
                                   "have been applied first", method)
                    self._shutdown_process()
                    raise
            logger.warning("Reverie: MCP call failed (%s); restarting the server and retrying", exc)
            self._shutdown_process()
            self.start()
            return self._request(method, params) or {}


def _content_text(result: Dict[str, Any]) -> str:
    """Concatenate the text blocks of an MCP tool result."""
    blocks = (result or {}).get("content") or []
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


def server_env_from_environ(extra: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """The NEO4J_*/REVERIE_* variables the server needs, from this process's environment.

    ``extra`` (typically plugin config) wins over the environment; None values are dropped.
    """
    env = {k: v for k, v in os.environ.items() if k.startswith(SERVER_ENV_PREFIXES)}
    for key, value in (extra or {}).items():
        if value is not None and str(value) != "":
            env[key] = str(value)
    return env
