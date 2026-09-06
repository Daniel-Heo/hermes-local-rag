"""Publication boundaries for local-only evaluation output."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('artifact', [
    'evaluation/visual-quality/native-run-01/ranked.sealed.json',
    'evaluation/relevance/run-02/report.json',
    'evaluation/visual-quality/runs/probe/metrics.json',
    'evaluation/relevance/results/report.json',
    'evaluation/relevance/attempt-01/response.txt',
    'evaluation/relevance/full-tests.xml',
    'evaluation/visual-quality/results/report.json',
])
def test_local_run_outputs_are_ignored(tmp_path, artifact):
    root = Path(__file__).resolve().parents[2]
    shutil.copyfile(root / '.gitignore', tmp_path / '.gitignore')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    result = subprocess.run(['git', 'check-ignore', '--no-index', artifact],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, artifact


def test_relevance_cli_accepts_explicit_service_home(tmp_path, monkeypatch):
    from evaluation import relevance_benchmark as benchmark

    class ClientReached(Exception):
        pass

    homes = []

    def client(home, **kwargs):
        homes.append(Path(home))
        raise ClientReached

    monkeypatch.setattr(benchmark, 'ROOT', tmp_path)
    monkeypatch.setattr(benchmark, 'ExistingClient', client)
    home = tmp_path / 'isolated-service'
    monkeypatch.setattr(sys, 'argv', ['benchmark', 'calibrate', '--run',
                                     str(tmp_path / 'evaluation' / 'run-test'),
                                     '--service-home', str(home)])
    with pytest.raises(ClientReached):
        benchmark.main()
    assert homes == [home]
