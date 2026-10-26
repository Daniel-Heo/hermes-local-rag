"""Windows compatibility shims for Hermes Local RAG.

On Linux and macOS every helper here falls through to the original POSIX
behaviour, so the patched plugin behaves exactly as upstream there.
On native Windows:
  - fcntl.flock is emulated with msvcrt.locking on a byte far past EOF,
    so lock files can still be read and written normally.
  - os.getuid() does not exist; os.stat() reports st_uid == 0, so 0 is used.
  - POSIX permission bits are not meaningful, so the 0o077 check is skipped.
  - AF_UNIX is replaced by a 127.0.0.1 TCP listener. The port is written to
    the old socket path. Every request is still authenticated by the
    existing random token file, which stays inside the user's profile.
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

IS_WINDOWS = os.name == "nt"
O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
# LOCAL_RAG_FORCE_TCP=1 exercises the Windows transport on POSIX for testing.
USE_AF_UNIX = (hasattr(socket, "AF_UNIX") and not IS_WINDOWS
               and not os.environ.get("LOCAL_RAG_FORCE_TCP"))


def getuid() -> int:
    return os.getuid() if hasattr(os, "getuid") else 0


def insecure_mode(st_mode: int) -> bool:
    return False if IS_WINDOWS else bool(st_mode & 0o077)


def fchmod(fd: int, mode: int) -> None:
    if not IS_WINDOWS and hasattr(os, "fchmod"):
        os.fchmod(fd, mode)


if IS_WINDOWS:
    import msvcrt

    class _FcntlShim:
        LOCK_SH = 1
        LOCK_EX = 2
        LOCK_NB = 4
        LOCK_UN = 8
        _OFFSET = 0x7FFFFFFF

        def flock(self, target, operation: int) -> None:
            fd = target if isinstance(target, int) else target.fileno()
            current = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, self._OFFSET, os.SEEK_SET)
                if operation & self.LOCK_UN:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                    return
                while True:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        return
                    except OSError as exc:
                        if operation & self.LOCK_NB:
                            raise BlockingIOError(exc.errno, "Lock is held by another process") from None
                        time.sleep(0.05)
            finally:
                os.lseek(fd, current, os.SEEK_SET)

    fcntl = _FcntlShim()
else:
    import fcntl  # noqa: F401


def listen_socket(socket_path: Path, *, backlog: int = 32) -> socket.socket:
    socket_path = Path(socket_path)
    if USE_AF_UNIX:
        listener = socket.socket(socket.AF_UNIX)
        try:
            listener.bind(str(socket_path))
            socket_path.chmod(0o600)
            listener.listen(backlog)
        except BaseException:
            listener.close()
            raise
        return listener
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(backlog)
        port = listener.getsockname()[1]
        temporary = socket_path.with_name(socket_path.name + ".tmp")
        temporary.write_text(str(port), encoding="ascii")
        os.replace(temporary, socket_path)
    except BaseException:
        listener.close()
        raise
    return listener


def connect_socket(socket_path: Path, timeout: float) -> socket.socket:
    socket_path = Path(socket_path)
    if USE_AF_UNIX:
        conn = socket.socket(socket.AF_UNIX)
        try:
            conn.settimeout(timeout)
            conn.connect(str(socket_path))
        except BaseException:
            conn.close()
            raise
        return conn
    text = socket_path.read_text(encoding="ascii").strip()  # FileNotFoundError when not running
    try:
        port = int(text)
    except ValueError:
        raise ConnectionRefusedError("Inference port file is incomplete") from None
    try:
        return socket.create_connection(("127.0.0.1", port), timeout=timeout)
    except (socket.timeout, TimeoutError):
        # Windows retries refused loopback connects for ~2 s; treat as not running.
        raise ConnectionRefusedError("Inference service did not accept the connection") from None


def default_hermes_home() -> str:
    """HERMES_HOME, else the platform default (%LOCALAPPDATA%\\hermes on Windows, ~/.hermes elsewhere)."""
    configured = os.environ.get("HERMES_HOME")
    if configured:
        return configured
    if IS_WINDOWS and os.environ.get("LOCALAPPDATA"):
        return os.path.join(os.environ["LOCALAPPDATA"], "hermes")
    return os.path.join(os.path.expanduser("~"), ".hermes")


def inference_python() -> str:
    """Interpreter for the inference child.

    On Windows the Hermes gateway runs inside a bundled runtime that does not
    have the Local RAG dependencies, so prefer the Hermes venv (HERMES_PYTHON).
    """
    if IS_WINDOWS:
        candidates = [os.environ.get("HERMES_PYTHON")]
        home = default_hermes_home()
        candidates.append(os.path.join(home, "hermes-agent", "venv", "Scripts", "python.exe"))
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return candidate
    return sys.executable


def launch_fd_args(fd: int) -> list[str]:
    return [] if IS_WINDOWS else ["--launch-fd", str(fd)]


def detached_popen_kwargs(fd: int) -> dict:
    if IS_WINDOWS:
        import subprocess
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"close_fds": True, "creationflags": flags}
    return {"start_new_session": True, "close_fds": True, "pass_fds": (fd,)}
