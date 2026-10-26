"""Windows compatibility layer: shared test helpers plus tests of every shim.

The shims are exercised on every platform where possible (TCP transport and
link rejection are forced on POSIX); Windows-only branches skip elsewhere.
"""
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from local_rag import _wincompat as wc
from local_rag._wincompat import fcntl

IS_WINDOWS = os.name == 'nt'
windows_only = pytest.mark.skipif(not IS_WINDOWS, reason='Windows-only shim branch')


def read_line(stream, timeout):
    """Readline with a deadline; select() only accepts sockets on Windows."""
    if not IS_WINDOWS:
        import select
        assert select.select([stream], [], [], timeout)[0], 'no output before deadline'
        return stream.readline()
    lines = queue.Queue()
    threading.Thread(target=lambda: lines.put(stream.readline()), daemon=True).start()
    try:
        return lines.get(timeout=timeout)
    except queue.Empty:
        raise AssertionError('no output before deadline') from None


def wait_output(process, timeout):
    """Wait until a child writes output (POSIX) or exits (Windows, no select on pipes)."""
    if not IS_WINDOWS:
        import select
        assert select.select([process.stdout, process.stderr], [], [], timeout)[0]
        return
    process.wait(timeout=timeout)


def pid_alive(pid):
    """os.kill(pid, 0) terminates the process on Windows, so query its exit code instead."""
    if not IS_WINDOWS:
        os.kill(pid, 0)
        return True
    import ctypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def wait_pid_exit(pid, timeout):
    deadline = time.monotonic() + timeout
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(.02)


def service_pid(process, ready_line):
    """PID announced by the service. A Windows venv python.exe is a launcher that runs
    the real interpreter as a child, so Popen.pid is not the service there."""
    pid = json.loads(ready_line)['pid']
    if not IS_WINDOWS:
        assert pid == process.pid
    return pid


def stop_profile_daemon(home, timeout=5):
    """Kill whatever daemon still answers for this profile and wait until it has exited."""
    from local_rag.inference import InferenceClient
    try:
        pid = InferenceClient(home, timeout=.2).request('ping')['pid']
    except (OSError, RuntimeError, ValueError):
        return
    os.kill(pid, HARD_KILL)
    wait_pid_exit(pid, timeout)


def remove_tree(path, timeout=5):
    """rmtree that waits for Windows handles (e.g. a dying venv launcher) to be released."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.1)


# A hard kill: SIGKILL on POSIX, TerminateProcess (any os.kill signal) on Windows.
HARD_KILL = getattr(signal, 'SIGKILL', signal.SIGTERM)


def graceful_popen_kwargs():
    """A new process group lets Windows tests send CTRL_BREAK_EVENT to only this child."""
    return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WINDOWS else {}


def stop_gracefully(process):
    """SIGTERM equivalent: Popen.terminate() is TerminateProcess on Windows and skips cleanup."""
    if IS_WINDOWS:
        os.kill(process.pid, signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()


# --- default_hermes_home / inference_python -------------------------------

def test_default_hermes_home_prefers_explicit_profile(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    assert wc.default_hermes_home() == str(tmp_path)


@windows_only
def test_default_hermes_home_uses_localappdata_on_windows(monkeypatch, tmp_path):
    monkeypatch.delenv('HERMES_HOME', raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    assert wc.default_hermes_home() == os.path.join(str(tmp_path), 'hermes')
    monkeypatch.delenv('LOCALAPPDATA')
    assert wc.default_hermes_home() == os.path.join(os.path.expanduser('~'), '.hermes')


@windows_only
def test_inference_python_prefers_hermes_venv_on_windows(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_PYTHON', raising=False)
    assert wc.inference_python() == sys.executable
    venv = tmp_path / 'hermes-agent' / 'venv' / 'Scripts' / 'python.exe'
    venv.parent.mkdir(parents=True)
    venv.write_text('')
    assert wc.inference_python() == str(venv)
    explicit = tmp_path / 'explicit.exe'
    explicit.write_text('')
    monkeypatch.setenv('HERMES_PYTHON', str(explicit))
    assert wc.inference_python() == str(explicit)
    monkeypatch.setenv('HERMES_PYTHON', str(tmp_path / 'missing.exe'))
    assert wc.inference_python() == str(venv)


def test_inference_python_is_current_interpreter_on_posix(monkeypatch, tmp_path):
    if IS_WINDOWS:
        pytest.skip('POSIX branch')
    monkeypatch.setenv('HERMES_PYTHON', str(tmp_path / 'ignored'))
    assert wc.inference_python() == sys.executable


# --- flock replacement ----------------------------------------------------

def test_flock_is_exclusive_nonblocking_and_keeps_file_usable(tmp_path):
    path = tmp_path / 'owner.lock'
    path.write_text('12.5')
    with path.open('r+') as first, path.open('r+') as second:
        fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The lock must not block normal reads/writes or move the file position.
        assert second.read() == '12.5'
        first.seek(0)
        first.truncate()
        first.write('7')
        first.flush()
        assert first.tell() == 1
        fcntl.flock(first, fcntl.LOCK_UN)
        fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(second, fcntl.LOCK_UN)
    assert path.read_text() == '7'


def test_flock_is_released_when_holder_process_exits(tmp_path):
    path = tmp_path / 'launch.lock'
    path.touch()
    code = ('import sys, time\nfrom local_rag._wincompat import fcntl\n'
            'h = open(sys.argv[1], "r+")\nfcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
            'print("locked", flush=True)\ntime.sleep(60)\n')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    child = subprocess.Popen([sys.executable, '-c', code, str(path)], stdout=subprocess.PIPE,
                             text=True, env=env, cwd=str(Path(wc.__file__).resolve().parent.parent))
    try:
        assert read_line(child.stdout, 10).strip() == 'locked'
        with path.open('r+') as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child.kill()
            child.wait(timeout=10)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


# --- TCP transport (forced on POSIX) --------------------------------------

@pytest.fixture
def tcp(monkeypatch):
    monkeypatch.setattr(wc, 'USE_AF_UNIX', False)


def test_tcp_listener_publishes_loopback_port_file(tmp_path, tcp):
    path = tmp_path / 'i.sock'
    listener = wc.listen_socket(path)
    try:
        host, port = listener.getsockname()[:2]
        assert host == '127.0.0.1' and path.read_text(encoding='ascii') == str(port)
        assert not path.with_name('i.sock.tmp').exists()
        with wc.connect_socket(path, 2) as conn:
            server, _ = listener.accept()
            with server:
                conn.sendall(b'ping\n')
                assert server.recv(16) == b'ping\n'
    finally:
        listener.close()
    # A stale port file from a dead service reads as "not running", never as a hang.
    with pytest.raises(ConnectionRefusedError):
        wc.connect_socket(path, .5).close()


def test_tcp_connect_reports_missing_and_incomplete_port_files(tmp_path, tcp):
    path = tmp_path / 'i.sock'
    with pytest.raises(FileNotFoundError):
        wc.connect_socket(path, .5)
    path.write_text('', encoding='ascii')
    with pytest.raises(ConnectionRefusedError, match='incomplete'):
        wc.connect_socket(path, .5)


def test_tcp_service_still_requires_profile_token(tmp_path, tcp):
    from local_rag.inference import InferenceClient, InferenceServer, _receive, _send
    server = InferenceServer(tmp_path, backend=lambda op, value, **kwargs: [1.])
    server.start()
    try:
        assert int(server.socket_path.read_text(encoding='ascii')) > 0
        with wc.connect_socket(server.socket_path, 5) as conn:
            _send(conn, {'token': 'wrong', 'op': 'ping'})
            assert 'Unauthorized' in _receive(conn)['error']
        assert InferenceClient(tmp_path).embed_query('ok') == [1.]
        assert InferenceClient(tmp_path).request('ping')['pid'] == os.getpid()
        with pytest.raises(RuntimeError, match='already running'):
            InferenceServer(tmp_path, backend=lambda *args, **kwargs: [1.]).start()
    finally:
        server.close()
    assert not server.socket_path.exists()
    assert not (server.directory / 'token').exists()


def test_tcp_transport_ignores_unix_socket_path_limit(tmp_path, tcp):
    from local_rag.inference import InferenceServer
    home = tmp_path / ('p' * 120)
    server = InferenceServer(home, backend=lambda *args, **kwargs: [1.])
    server.start()
    server.close()


# --- O_NOFOLLOW replacement -----------------------------------------------

def _symlink_or_skip(link, target, **kwargs):
    try:
        link.symlink_to(target, **kwargs)
    except OSError as exc:
        pytest.skip(f'Symlink creation not permitted here: {exc}')


def test_open_nofollow_opens_and_creates_regular_files(tmp_path):
    path = tmp_path / 'regular'
    fd = wc.open_nofollow(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, 'r+') as handle:
        handle.write('ok')
    fd = wc.open_nofollow(path, os.O_RDONLY)
    with os.fdopen(fd) as handle:
        assert handle.read() == 'ok'


def test_open_nofollow_rejects_file_symlink_without_touching_target(tmp_path):
    victim = tmp_path / 'victim'
    victim.write_text('keep')
    link = tmp_path / 'link'
    _symlink_or_skip(link, victim)
    for flags in (os.O_RDONLY, os.O_CREAT | os.O_RDWR):
        with pytest.raises(OSError):
            wc.open_nofollow(link, flags, 0o600)
    assert victim.read_text() == 'keep'


def test_open_nofollow_rejects_dangling_symlink_without_creating_target(tmp_path):
    link = tmp_path / 'link'
    _symlink_or_skip(link, tmp_path / 'created-through-link')
    with pytest.raises(OSError):
        wc.open_nofollow(link, os.O_CREAT | os.O_RDWR, 0o600)
    assert not (tmp_path / 'created-through-link').exists()


# --- graceful stop ----------------------------------------------------------

def test_stop_signals_add_ctrl_break_only_on_windows():
    assert wc.stop_signals() == ((signal.SIGBREAK,) if IS_WINDOWS else ())


# --- launch.lock handoff (no-sharing handle) --------------------------------

@windows_only
def test_unshared_launch_guard_survives_starter_close_via_inherited_handle(tmp_path):
    path = tmp_path / 'launch.lock'
    fd = wc.open_unshared(path)
    with pytest.raises(BlockingIOError):
        wc.open_unshared(path)
    holder = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.readline()'],
                              stdin=subprocess.PIPE, **wc.detached_popen_kwargs(fd))
    try:
        os.close(fd)  # Starter gives up its handle; the child's inherited one keeps the guard.
        with pytest.raises(BlockingIOError):
            wc.open_unshared(path)
    finally:
        holder.communicate(b'\n', timeout=5)
    os.close(wc.open_unshared(path))  # Holder exit released it.


@windows_only
def test_launch_fd_args_pass_os_handle_released_by_close_launch_fd(tmp_path):
    import _winapi
    import msvcrt
    path = tmp_path / 'launch.lock'
    fd = wc.open_unshared(path)
    assert wc.launch_fd_args(fd) == ['--launch-fd', str(msvcrt.get_osfhandle(fd))]
    # A raw duplicate stands in for the daemon's inherited handle value.
    process = _winapi.GetCurrentProcess()
    handle = _winapi.DuplicateHandle(process, msvcrt.get_osfhandle(fd), process, 0, False,
                                     _winapi.DUPLICATE_SAME_ACCESS)
    os.close(fd)
    with pytest.raises(BlockingIOError):
        wc.open_unshared(path)
    wc.close_launch_fd(handle)
    os.close(wc.open_unshared(path))


# --- host-runtime fallback of inference_python() --------------------------

def _fallback_home(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_PYTHON', raising=False)
    monkeypatch.delenv('HERMES_EMBEDDINGGEMMA_DIR', raising=False)
    assert wc.inference_python() == sys.executable


@windows_only
def test_fallback_runtime_without_dependencies_is_unavailable_with_reason(monkeypatch, tmp_path):
    import local_rag.provider as provider
    _fallback_home(monkeypatch, tmp_path)
    model = tmp_path / 'models' / 'embeddinggemma-litert' / 'embeddinggemma-300M_seq512_mixed-precision.tflite'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'')
    # A missing module stands in for the Hermes bundled runtime lacking numpy/litert.
    monkeypatch.setattr(provider, '_INFERENCE_MODULES', (*provider._INFERENCE_MODULES, 'local_rag_missing_dep'))
    with pytest.warns(RuntimeWarning, match='HERMES_PYTHON'):
        assert provider.LocalRagProvider().is_available() is False


@windows_only
def test_fallback_runtime_import_failure_reports_clear_start_error(monkeypatch, tmp_path):
    import local_rag.inference as inference
    _fallback_home(monkeypatch, tmp_path)
    real_popen = subprocess.Popen
    def missing_dependency(argv, **kwargs):
        assert argv[0] == sys.executable
        return real_popen([sys.executable, '-c', "raise ModuleNotFoundError(\"No module named 'ai_edge_litert'\")"],
                          **kwargs)
    monkeypatch.setattr(inference.subprocess, 'Popen', missing_dependency)
    with pytest.raises(RuntimeError, match='failed to start; see private inference.log'):
        inference.InferenceClient(tmp_path, startup_timeout=10).ensure_started()
    log = (tmp_path / 'local-rag' / '.cache' / 'inference.log').read_text()
    assert "No module named 'ai_edge_litert'" in log


# --- link detection for managed directories --------------------------------

def _junction(link, target):
    import _winapi
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    _winapi.CreateJunction(str(target), str(link))


def test_is_link_matches_is_symlink_for_regular_and_missing_paths(tmp_path):
    (tmp_path / 'dir').mkdir()
    (tmp_path / 'file').write_text('')
    for path in (tmp_path / 'dir', tmp_path / 'file', tmp_path / 'missing'):
        assert wc.is_link(path) is False
    link = tmp_path / 'link'
    _symlink_or_skip(link, tmp_path / 'dir', target_is_directory=True)
    assert wc.is_link(link) is True


@windows_only
def test_is_link_rejects_junction_that_is_symlink_misses(tmp_path):
    link = tmp_path / 'junction'
    _junction(link, tmp_path / 'elsewhere')
    assert not link.is_symlink()  # The gap the shim closes.
    assert wc.is_link(link) is True


@windows_only
def test_curated_images_reject_junctioned_directories(tmp_path):
    from local_rag.images import CuratedImages
    hijacked = tmp_path / 'hijacked'
    _junction(hijacked / 'local-rag' / 'images', tmp_path / 'outside-images')
    with pytest.raises(ValueError, match='must not be a symlink'):
        CuratedImages(hijacked, text=None, visual=None)
    images = CuratedImages(tmp_path / 'home', text=None, visual=None)
    _junction(tmp_path / 'home' / 'local-rag' / '.cache', tmp_path / 'outside-cache')
    with pytest.raises(ValueError, match='staging directory must not be a symlink'):
        images._private_cache()


# --- evaluation CLI default service home ------------------------------------

@windows_only
def test_relevance_cli_defaults_service_home_to_windows_hermes_home(tmp_path, monkeypatch):
    from evaluation import relevance_benchmark as benchmark

    class ClientReached(Exception):
        pass

    homes = []

    def client(home, **kwargs):
        homes.append(Path(home))
        raise ClientReached

    monkeypatch.setattr(benchmark, 'ROOT', tmp_path)
    monkeypatch.setattr(benchmark, 'ExistingClient', client)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'service'))
    monkeypatch.setattr(sys, 'argv', ['benchmark', 'calibrate', '--run', str(tmp_path / 'evaluation' / 'run-test')])
    with pytest.raises(ClientReached):
        benchmark.main()
    assert homes == [(tmp_path / 'service').resolve()]
    monkeypatch.delenv('HERMES_HOME')
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'appdata'))
    with pytest.raises(ClientReached):
        benchmark.main()
    assert homes[1] == (tmp_path / 'appdata' / 'hermes').resolve()


@windows_only
def test_curated_images_treat_junctioned_managed_file_as_link(tmp_path):
    from local_rag.tests.test_image_reconciliation import make_images, save
    images, source = make_images(tmp_path)
    try:
        managed = Path(save(images, source)['path'])
        managed.unlink()
        outside = tmp_path / 'outside'
        _junction(managed, outside)
        (outside / 'keep').write_text('keep')
        with pytest.raises(ValueError, match='integrity failure'):
            save(images, source, group='other')  # New row reusing a managed name that is a link.
        images.reconcile('owner', force=True)  # _reconcile_row drops the linked row.
        assert images.count('owner') == 0
        assert (outside / 'keep').read_text() == 'keep'
    finally:
        images.close()


@windows_only
def test_inference_rejects_junctioned_private_directory(tmp_path):
    from local_rag.inference import InferenceClient, _private_directory
    home = tmp_path / 'home'
    _junction(home / 'local-rag' / '.cache', tmp_path / 'outside')
    with pytest.raises(PermissionError, match='not a symlink'):
        _private_directory(home / 'local-rag' / '.cache')
    with pytest.raises(PermissionError, match='Insecure inference directory'):
        InferenceClient(home)._request('ping')
    assert not list((tmp_path / 'outside').iterdir())
