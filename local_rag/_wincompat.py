"""Windows compatibility shims for Hermes Local RAG.

On Linux and macOS every helper here falls through to the original POSIX
behaviour, so the patched plugin behaves exactly as upstream there.
On native Windows:
  - fcntl.flock is emulated with msvcrt.locking on a byte far past EOF,
    so lock files can still be read and written normally.
  - os.getuid() does not exist; os.stat() reports st_uid == 0, so 0 is used.
  - POSIX permission bits are not meaningful, so the 0o077 check and the
    st_uid check become an owner + DACL check (acl_insecure), and the
    0o600/0o700 chmods become a protected DACL for this user, SYSTEM and
    Administrators (the trust set Windows OpenSSH uses for private keys).
  - O_NOFOLLOW does not exist; open_nofollow() rejects symlinks and junctions.
  - AF_UNIX is replaced by a 127.0.0.1 TCP listener. The port is written to
    the old socket path. Every request is still authenticated by the
    existing random token file, which stays inside the user's profile.
"""
from __future__ import annotations

import errno
import os
import signal
import socket
import stat
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


def insecure_mode(st_mode: int, target=None) -> bool:
    """Group/other mode bits; on Windows the owner and DACL of target (an fd or a path)."""
    return acl_insecure(target) if IS_WINDOWS else bool(st_mode & 0o077)


def fchmod(fd: int, mode: int) -> None:
    if IS_WINDOWS:
        if not mode & 0o077 and acl_insecure(fd):
            _restrict(fd, directory=False)
    elif hasattr(os, "fchmod"):
        os.fchmod(fd, mode)


def chmod(path, mode: int) -> None:
    os.chmod(path, mode)
    if IS_WINDOWS and not mode & 0o077 and acl_insecure(path):
        _restrict(path, directory=os.path.isdir(path))


def foreign_owner(path) -> bool:
    """Windows replacement for st_uid != getuid(), which is always 0 == 0 there."""
    return IS_WINDOWS and _read_security(path)[0] not in _trusted_sids()


def _is_link(info: os.stat_result) -> bool:
    # Symlinks and junctions are both reparse points on Windows.
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def is_link_stat(info: os.stat_result) -> bool:
    """For lstat() results: S_ISLNK, plus junctions (which lstat reports as S_ISDIR) on Windows."""
    return _is_link(info)


def is_link(path) -> bool:
    """Path.is_symlink(); on Windows junctions (and any reparse point) count too."""
    if not IS_WINDOWS:
        return Path(path).is_symlink()
    try:
        return _is_link(os.lstat(path))
    except (OSError, ValueError):
        return False


def open_nofollow(path, flags: int, mode: int = 0o777) -> int:
    """os.open with O_NOFOLLOW; Windows has no O_NOFOLLOW, so links are rejected explicitly."""
    if not IS_WINDOWS:
        return os.open(path, flags | O_NOFOLLOW, mode)
    try:
        if _is_link(os.lstat(path)):
            raise OSError(errno.ELOOP, "Refusing to follow a link", str(path))
    except FileNotFoundError:
        pass
    created = None
    if flags & os.O_CREAT and not mode & 0o077:
        # Like a POSIX create with 0o600: the file is private from birth.
        created = _create_private_file(path, exclusive=bool(flags & os.O_EXCL))
        if created is not None:
            flags &= ~(os.O_CREAT | os.O_EXCL)
    fd = os.open(path, flags, mode)
    try:
        # Reject a link swapped in between the check and the open.
        after, opened = os.lstat(path), os.fstat(fd)
        if _is_link(after) or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError(errno.ELOOP, "Refusing to follow a link", str(path))
        if created is not None and created != (opened.st_dev, opened.st_ino):
            raise OSError(errno.EEXIST, "Private file was replaced after creation", str(path))
    except BaseException:
        os.close(fd)
        raise
    return fd


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


def shared_inference_home(home) -> Path:
    """Home whose inference service and models every Hermes profile shares.

    Embedding is stateless, so one daemon serves all profiles; memory databases stay per profile.
    LOCAL_RAG_INFERENCE_HOME wins; a profile home (<root>/profiles/<name>) maps to <root>.
    """
    configured = os.environ.get("LOCAL_RAG_INFERENCE_HOME")
    path = Path(configured or home).expanduser().resolve()
    if not configured and path.parent.name == "profiles":
        return path.parent.parent
    return path


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


def detached_interpreter(python: str, env: dict) -> str:
    """Executable for a detached child that inherits the launch guard handle.

    A Windows venv's Scripts\\python.exe is a launcher that respawns the base
    interpreter and keeps every inherited handle for the child's lifetime, so
    the guard would never be released. Start the base interpreter directly
    with __PYVENV_LAUNCHER__, exactly as the launcher does, so sys.prefix,
    sys.executable and sys.path stay those of the venv. Updates env in place.
    """
    if not IS_WINDOWS:
        return python
    # A client started through a venv launcher must not leak its venv to the child.
    env.pop("__PYVENV_LAUNCHER__", None)
    launcher = Path(python)
    config = launcher.parent.parent / "pyvenv.cfg"
    if launcher.parent.name.lower() != "scripts" or not config.is_file():
        return python
    home = None
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip().lower() == "home":
            home = value.strip()
    base = Path(home) / launcher.name if home else None
    if base is None or not base.is_file() or base == launcher:
        return python
    env["__PYVENV_LAUNCHER__"] = str(launcher)
    return str(base)


def stop_signals() -> tuple:
    """Extra graceful-stop signals: SIGBREAK (CTRL_BREAK_EVENT) on Windows."""
    return (signal.SIGBREAK,) if IS_WINDOWS else ()


def open_unshared(path) -> int:
    """Windows-only launch guard: open with share mode 0, links rejected.

    A byte-range lock dies with the locking process's handle, so it cannot be
    handed across process creation. A no-sharing open can: the file object
    stays exclusive while any inherited handle to it is open, so every other
    open fails until the daemon (or whatever inherited it) closes it or exits.
    Raises BlockingIOError while another holder exists.
    """
    import _winapi
    import msvcrt
    try:
        if _is_link(os.lstat(path)):
            raise OSError(errno.ELOOP, "Refusing to follow a link", str(path))
    except FileNotFoundError:
        pass
    try:
        handle = _winapi.CreateFile(str(path), _winapi.GENERIC_READ | _winapi.GENERIC_WRITE, 0, 0,
                                    _OPEN_ALWAYS, _FILE_FLAG_OPEN_REPARSE_POINT, 0)
    except OSError as exc:
        if getattr(exc, "winerror", None) == _ERROR_SHARING_VIOLATION:
            raise BlockingIOError(errno.EAGAIN, "Launch guard is held by another process") from None
        raise
    fd = msvcrt.open_osfhandle(handle, os.O_RDWR)
    try:
        after, opened = os.lstat(path), os.fstat(fd)
        if _is_link(after) or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError(errno.ELOOP, "Refusing to follow a link", str(path))
    except BaseException:
        os.close(fd)
        raise
    return fd


_OPEN_ALWAYS = 4
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_ERROR_SHARING_VIOLATION = 32


def launch_fd_args(fd: int) -> list[str]:
    # Windows passes the inherited OS handle value, which is identical in the child.
    if IS_WINDOWS:
        import msvcrt
        return ["--launch-fd", str(msvcrt.get_osfhandle(fd))]
    return ["--launch-fd", str(fd)]


def close_launch_fd(value: int) -> None:
    if IS_WINDOWS:
        import _winapi
        _winapi.CloseHandle(value)
    else:
        os.close(value)


def detached_popen_kwargs(fd: int) -> dict:
    if IS_WINDOWS:
        import msvcrt
        import subprocess
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # Inherit only the launch guard (plus std handles), like pass_fds on POSIX.
        handle = msvcrt.get_osfhandle(fd)
        os.set_handle_inheritable(handle, True)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.lpAttributeList = {"handle_list": [handle]}
        return {"close_fds": True, "creationflags": flags, "startupinfo": startupinfo}
    return {"start_new_session": True, "close_fds": True, "pass_fds": (fd,)}


# --- Windows ACLs: the replacement for 0o600/0o700 and the 0o077 check ------
#
# A private file or directory may grant read/write class rights only to its
# owner, this user, LocalSystem and BUILTIN\Administrators, and must be owned
# by one of the latter three. This is the trust set Windows OpenSSH requires
# for private keys (contrib/win32/win32compat/w32-sshfileperm.c,
# check_secure_file_permission). A NULL DACL grants everyone full access.

_SE_FILE_OBJECT = 1
_OWNER_SECURITY_INFORMATION = 0x00000001
_DACL_SECURITY_INFORMATION = 0x00000004
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
_ACCESS_ALLOWED_ACE_TYPE = 0
_ACCESS_DENIED_ACE_TYPE = 1
# Read/write class rights: data, EA, attributes, child delete, delete, DACL,
# owner and the generic read/write/all bits. Traverse and synchronize are harmless.
_FOREIGN_ACCESS_MASK = (0x00000001 | 0x00000002 | 0x00000004 | 0x00000010 | 0x00000040
                        | 0x00000100 | 0x00010000 | 0x00040000 | 0x00080000
                        | 0x10000000 | 0x40000000 | 0x80000000)
_SYSTEM_SID = "S-1-5-18"
_ADMINISTRATORS_SID = "S-1-5-32-544"
# CREATOR OWNER and OWNER RIGHTS stand for the (already trusted) owner.
_OWNER_ALIAS_SIDS = frozenset({"S-1-3-0", "S-1-3-4"})
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_READ_CONTROL = 0x00020000
_WRITE_DAC = 0x00040000
_FILE_READ_ATTRIBUTES = 0x0080
_FILE_SHARE_ALL = 0x1 | 0x2 | 0x4
_CREATE_NEW = 1
_FILE_ATTRIBUTE_NORMAL = 0x80
_ERROR_FILE_EXISTS = 80
_ERROR_ALREADY_EXISTS = 183

if IS_WINDOWS:
    import ctypes
    import functools
    from ctypes import wintypes

    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _INVALID_HANDLE = ctypes.c_void_p(-1).value
    _VOIDP = ctypes.c_void_p
    _PVOIDP = ctypes.POINTER(ctypes.c_void_p)

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p),
                    ("bInheritHandle", wintypes.BOOL)]

    class _AclHeader(ctypes.Structure):
        _fields_ = [("AclRevision", ctypes.c_ubyte), ("Sbz1", ctypes.c_ubyte), ("AclSize", wintypes.WORD),
                    ("AceCount", wintypes.WORD), ("Sbz2", wintypes.WORD)]

    class _AceHeader(ctypes.Structure):
        # ACE_HEADER plus the Mask of ACCESS_ALLOWED/DENIED_ACE; SidStart follows.
        _fields_ = [("AceType", ctypes.c_ubyte), ("AceFlags", ctypes.c_ubyte), ("AceSize", wintypes.WORD),
                    ("Mask", wintypes.DWORD)]

    def _bind(library, name, restype, *argtypes):
        function = getattr(library, name)
        function.restype, function.argtypes = restype, list(argtypes)
        return function

    _GetSecurityInfo = _bind(_advapi32, "GetSecurityInfo", wintypes.DWORD, wintypes.HANDLE, ctypes.c_int,
                             wintypes.DWORD, _PVOIDP, _PVOIDP, _PVOIDP, _PVOIDP, _PVOIDP)
    _GetNamedSecurityInfoW = _bind(_advapi32, "GetNamedSecurityInfoW", wintypes.DWORD, wintypes.LPCWSTR,
                                   ctypes.c_int, wintypes.DWORD, _PVOIDP, _PVOIDP, _PVOIDP, _PVOIDP, _PVOIDP)
    _SetSecurityInfo = _bind(_advapi32, "SetSecurityInfo", wintypes.DWORD, wintypes.HANDLE, ctypes.c_int,
                             wintypes.DWORD, _VOIDP, _VOIDP, _VOIDP, _VOIDP)
    _SetNamedSecurityInfoW = _bind(_advapi32, "SetNamedSecurityInfoW", wintypes.DWORD, wintypes.LPCWSTR,
                                   ctypes.c_int, wintypes.DWORD, _VOIDP, _VOIDP, _VOIDP, _VOIDP)
    _GetAce = _bind(_advapi32, "GetAce", wintypes.BOOL, _VOIDP, wintypes.DWORD, _PVOIDP)
    _ConvertSidToStringSidW = _bind(_advapi32, "ConvertSidToStringSidW", wintypes.BOOL, _VOIDP, _PVOIDP)
    _ConvertStringSecurityDescriptorToSecurityDescriptorW = _bind(
        _advapi32, "ConvertStringSecurityDescriptorToSecurityDescriptorW", wintypes.BOOL, wintypes.LPCWSTR,
        wintypes.DWORD, _PVOIDP, _VOIDP)
    _GetSecurityDescriptorDacl = _bind(_advapi32, "GetSecurityDescriptorDacl", wintypes.BOOL, _VOIDP,
                                       ctypes.POINTER(wintypes.BOOL), _PVOIDP, ctypes.POINTER(wintypes.BOOL))
    _OpenProcessToken = _bind(_advapi32, "OpenProcessToken", wintypes.BOOL, wintypes.HANDLE, wintypes.DWORD,
                              ctypes.POINTER(wintypes.HANDLE))
    _GetTokenInformation = _bind(_advapi32, "GetTokenInformation", wintypes.BOOL, wintypes.HANDLE, ctypes.c_int,
                                 _VOIDP, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
    _GetCurrentProcess = _bind(_kernel32, "GetCurrentProcess", wintypes.HANDLE)
    _CloseHandle = _bind(_kernel32, "CloseHandle", wintypes.BOOL, wintypes.HANDLE)
    _LocalFree = _bind(_kernel32, "LocalFree", _VOIDP, _VOIDP)
    _CreateFileW = _bind(_kernel32, "CreateFileW", wintypes.HANDLE, wintypes.LPCWSTR, wintypes.DWORD,
                         wintypes.DWORD, ctypes.POINTER(_SecurityAttributes), wintypes.DWORD, wintypes.DWORD,
                         wintypes.HANDLE)
    _ReOpenFile = _bind(_kernel32, "ReOpenFile", wintypes.HANDLE, wintypes.HANDLE, wintypes.DWORD,
                        wintypes.DWORD, wintypes.DWORD)

    def _last_error():
        return ctypes.WinError(ctypes.get_last_error())

    def _sid_string(sid) -> str:
        text = ctypes.c_void_p()
        if not _ConvertSidToStringSidW(sid, ctypes.byref(text)):
            raise _last_error()
        try:
            return ctypes.wstring_at(text.value)
        finally:
            _LocalFree(text)

    @functools.lru_cache(maxsize=None)
    def current_user_sid() -> str:
        token = wintypes.HANDLE()
        if not _OpenProcessToken(_GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)):
            raise _last_error()
        try:
            size = wintypes.DWORD()
            _GetTokenInformation(token, _TOKEN_USER, None, 0, ctypes.byref(size))
            buffer = ctypes.create_string_buffer(size.value)
            if not _GetTokenInformation(token, _TOKEN_USER, buffer, size, ctypes.byref(size)):
                raise _last_error()
            # TOKEN_USER starts with SID_AND_ATTRIBUTES.Sid.
            return _sid_string(ctypes.cast(buffer, _PVOIDP)[0])
        finally:
            _CloseHandle(token)


def _trusted_sids() -> frozenset:
    return frozenset({current_user_sid(), _SYSTEM_SID, _ADMINISTRATORS_SID})


def _read_security(target):
    """(owner SID, DACL entries or None for a NULL DACL) of an fd or a path.

    Each entry is (ace type, ace flags, access mask, SID or None for other ACE types).
    """
    owner, dacl, descriptor = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    information = _OWNER_SECURITY_INFORMATION | _DACL_SECURITY_INFORMATION
    if isinstance(target, int):
        error = _GetSecurityInfo(msvcrt.get_osfhandle(target), _SE_FILE_OBJECT, information,
                                 ctypes.byref(owner), None, ctypes.byref(dacl), None, ctypes.byref(descriptor))
    else:
        error = _GetNamedSecurityInfoW(str(target), _SE_FILE_OBJECT, information,
                                       ctypes.byref(owner), None, ctypes.byref(dacl), None,
                                       ctypes.byref(descriptor))
    if error:
        raise ctypes.WinError(error)
    try:
        entries = None
        if dacl.value:
            entries = []
            for index in range(_AclHeader.from_address(dacl.value).AceCount):
                ace = ctypes.c_void_p()
                if not _GetAce(dacl, index, ctypes.byref(ace)):
                    raise _last_error()
                header = _AceHeader.from_address(ace.value)
                sid = (_sid_string(ace.value + ctypes.sizeof(_AceHeader))
                       if header.AceType in (_ACCESS_ALLOWED_ACE_TYPE, _ACCESS_DENIED_ACE_TYPE) else None)
                entries.append((header.AceType, header.AceFlags, header.Mask, sid))
        return _sid_string(owner), entries
    finally:
        _LocalFree(descriptor)


def acl_insecure(target) -> bool:
    """Windows replacement for st_mode & 0o077 (and st_uid): see the section comment."""
    owner, entries = _read_security(target)
    trusted = _trusted_sids()
    if owner not in trusted or entries is None:
        return True
    for kind, _flags, mask, sid in entries:
        if kind == _ACCESS_DENIED_ACE_TYPE:
            continue
        if kind != _ACCESS_ALLOWED_ACE_TYPE:
            return True  # Conditional/callback grants cannot be judged here.
        if sid not in trusted and sid not in _OWNER_ALIAS_SIDS and mask & _FOREIGN_ACCESS_MASK:
            return True
    return False


def _private_descriptor(directory: bool):
    """Protected DACL: full control for this user, SYSTEM and Administrators only. LocalFree it."""
    inherit = "OICI" if directory else ""
    sddl = "D:P" + "".join(f"(A;{inherit};FA;;;{sid})" for sid in (current_user_sid(), "SY", "BA"))
    descriptor = ctypes.c_void_p()
    if not _ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(descriptor), None):
        raise _last_error()
    return descriptor


def _restrict(target, *, directory: bool) -> None:
    """Replace the DACL of an fd or a path with the private one, ending inheritance."""
    descriptor = _private_descriptor(directory)
    try:
        present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
        if not _GetSecurityDescriptorDacl(descriptor, ctypes.byref(present), ctypes.byref(dacl),
                                          ctypes.byref(defaulted)):
            raise _last_error()
        information = _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION
        if isinstance(target, int):
            # The fd's own handle lacks WRITE_DAC; reopen the same file object for it.
            handle = _ReOpenFile(msvcrt.get_osfhandle(target), _WRITE_DAC | _READ_CONTROL, _FILE_SHARE_ALL, 0)
            if handle in (None, _INVALID_HANDLE):
                raise _last_error()
            try:
                error = _SetSecurityInfo(handle, _SE_FILE_OBJECT, information, None, None, dacl, None)
            finally:
                _CloseHandle(handle)
        else:
            error = _SetNamedSecurityInfoW(str(target), _SE_FILE_OBJECT, information, None, None, dacl, None)
        if error:
            raise ctypes.WinError(error)
    finally:
        _LocalFree(descriptor)


def _create_private_file(path, *, exclusive: bool):
    """Create path with the private DACL atomically; (st_dev, st_ino), or None if it existed."""
    descriptor = _private_descriptor(False)
    try:
        attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
        handle = _CreateFileW(str(path), _FILE_READ_ATTRIBUTES, _FILE_SHARE_ALL, ctypes.byref(attributes),
                              _CREATE_NEW, _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT, None)
        if handle in (None, _INVALID_HANDLE):
            error = ctypes.get_last_error()
            if error in (_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS):
                if exclusive:
                    raise FileExistsError(errno.EEXIST, "File exists", str(path))
                return None
            raise ctypes.WinError(error)
    finally:
        _LocalFree(descriptor)
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    try:
        info = os.fstat(fd)
    finally:
        os.close(fd)
    return info.st_dev, info.st_ino
