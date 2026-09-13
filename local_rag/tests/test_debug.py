from __future__ import annotations

import json
import pytest

from local_rag.config import LocalRagConfig
from local_rag.provider import register


class Embedder:
    dimensions = 3

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    embed_document = embed_query


@pytest.fixture
def native_provider(tmp_path, monkeypatch):
    plugins = pytest.importorskip('hermes_cli.plugins')
    memory = pytest.importorskip('plugins.memory')
    constants = pytest.importorskip('hermes_constants')
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, '_plugin_manager', manager)
    token = constants.set_hermes_home_override(str(tmp_path))
    collector = memory._ProviderCollector()
    register(collector)
    provider = collector.provider
    provider._embedder = Embedder()
    LocalRagConfig(debug_enabled=True).save(tmp_path)
    (tmp_path / 'project').mkdir()
    provider.initialize('debug-session', hermes_home=str(tmp_path), cwd=str(tmp_path / 'project'))
    try:
        yield provider, manager, tmp_path
    finally:
        provider.shutdown()
        constants.reset_hermes_home_override(token)


def save(provider, text='The project uses Python for indexing.'):
    return json.loads(provider.handle_tool_call('local_rag_remember', {
        'text': text, 'scope': 'project', 'subject': 'indexing', 'durability': 'stable',
    }))


def finish(manager, session='debug-session'):
    from hermes_cli.lifecycle import invoke_hook
    return invoke_hook('transform_llm_output', response_text='Ответ.', session_id=session)


def test_real_memory_loader_connects_to_native_output_hook(native_provider):
    provider, manager, _ = native_provider
    provider.on_turn_start(1, 'Not retained by debug')
    assert save(provider)['stored']
    results = finish(manager)
    assert any('диагностика' in text and 'Python' in text for text in results)


def test_off_turn_does_not_capture_or_claim_output(native_provider):
    provider, manager, home = native_provider
    LocalRagConfig(debug_enabled=False).save(home)
    provider.on_turn_start(1, 'Private unretained conversation')
    assert save(provider)['stored']
    provider.prefetch('Python')
    assert not provider._debug._turns
    assert finish(manager) == []


def test_correction_and_forget_report_confirmed_actions_without_repeating_forgotten_text(native_provider):
    provider, manager, _ = native_provider
    provider.on_turn_start(1, '')
    assert save(provider)['stored']
    finish(manager)
    item = json.loads(provider.handle_tool_call('local_rag_search', {'query': 'Python'}))['results'][0]
    provider.on_turn_start(2, '')
    update = provider.handle_tool_call('local_rag_update', {
        'id': item['id'], 'expected_text': item['text'],
        'new_text': 'The project uses Rust for indexing.', 'subject': 'indexing', 'durability': 'stable',
    })
    assert json.loads(update)['updated']
    assert 'Исправлено' in finish(manager)[0]
    provider.on_turn_start(3, '')
    provider.prefetch('Rust')
    assert json.loads(provider.handle_tool_call('local_rag_forget', {'id': item['id']}))['removed']
    output = finish(manager)[0]
    assert 'Удалено' in output
    assert 'Rust' not in output
    assert provider._store.count(provider._namespace) == 0


def test_trace_is_bounded_and_cannot_escape_markup():
    from local_rag.debug import DebugTurns
    turns = DebugTurns()
    turns.start('s')
    turns.recall('s', '```\n</details><img src="https://invalid.test/pixel">\n' * 200)
    for _ in range(500):
        turns.change('s', 'A' * 2000, 'project')
        turns.failure('s', 'B' * 2000)
    output = turns.append('s', 'Ответ.', enabled=True)
    assert len(output) < 6500
    assert '</details>' not in output
    assert output.count('**Local RAG: диагностика**') == 1
    assert '<img' not in output
    assert 'сокращ' in output


def test_diagnostics_re_screen_legacy_recall_and_explicit_search_credentials():
    from local_rag.debug import DebugTurns
    turns = DebugTurns()
    secret = 'synthetic-opaque-credential'
    turns.start('recall')
    turns.recall('recall', f'ordinary record\nAuthorization:\nBearer {secret}')
    recall_output = turns.append('recall', 'Ответ.', enabled=True)
    assert secret not in recall_output
    assert 'скрыт фильтром безопасности' in recall_output
    turns.start('search')
    turns.search('search', [{'id': 1, 'text': f'Authorization: Bearer {secret}'}])
    search_output = turns.append('search', 'Ответ.', enabled=True)
    assert secret not in search_output
    assert 'содержимое скрыто фильтром безопасности' in search_output


def test_ambiguous_index_outcomes_do_not_claim_a_save():
    from local_rag.provider import LocalRagProvider
    provider = LocalRagProvider(embedder=Embedder())
    provider._debug.start('s')
    provider._trace_other_tool('local_rag_index_file', {}, {'chunks': 0}, 's')
    provider._trace_other_tool('local_rag_index_image', {'scope': 'project'}, {'id': 7}, 's')
    output = provider._debug.append('s', 'Ответ.', enabled=True)
    assert 'Наблюдаемые результаты' in output
    assert 'Сохранений нет' not in output
    assert 'Изображение сохранено' not in output


def test_explicit_search_and_document_import_are_visible(native_provider):
    provider, manager, home = native_provider
    path = home / 'project' / 'notes.md'
    path.write_text('The Cedar project reviews documents every Tuesday.')
    provider.on_turn_start(1, '')
    result = json.loads(provider.handle_tool_call('local_rag_index_file', {'path': str(path)}))
    assert result['chunks'] > 0
    assert 'Проиндексировано' in finish(manager)[0]
    provider.on_turn_start(2, '')
    result = json.loads(provider.handle_tool_call('local_rag_search', {'query': 'Cedar Tuesday'}))
    assert result['results']
    output = finish(manager)[0]
    assert 'Явный поиск' in output and 'Tuesday' in output


def test_failure_does_not_echo_exception_and_is_not_a_rejection(native_provider, monkeypatch):
    provider, manager, _ = native_provider
    provider.on_turn_start(1, '')
    def fail(text):
        raise RuntimeError('private exception material')
    monkeypatch.setattr(provider._embedder, 'embed_document', fail)
    assert 'error' in save(provider)
    output = finish(manager)[0]
    assert 'Сбой' in output and 'private exception material' not in output
    assert 'Сохранено:' not in output


def test_foreign_profile_session_and_completed_turn_do_not_get_trace(native_provider, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    provider, manager, _ = native_provider
    provider.on_turn_start(1, '')
    assert save(provider)['stored']
    assert finish(manager, 'another-session') == []
    token = set_hermes_home_override(str(tmp_path / 'other-home'))
    try:
        assert finish(manager) == []
    finally:
        reset_hermes_home_override(token)
    assert 'Python' in finish(manager)[0]
    assert finish(manager) == []
    provider.on_turn_start(2, '')
    assert save(provider, 'The project uses Rust for indexing.')['stored']
    manager.invoke_hook('on_session_end', session_id='debug-session')
    assert finish(manager) == []


def test_hot_toggle_does_not_reload_model_settings(native_provider):
    provider, manager, home = native_provider
    before = provider._config.embedding_dimensions
    LocalRagConfig(debug_enabled=True, embedding_dimensions=768).save(home)
    provider.on_turn_start(1, '')
    assert provider._config.embedding_dimensions == before
    assert 'Сохранений нет' in finish(manager)[0]


def test_builtin_mirror_writes_are_not_reported_as_no_save(native_provider):
    provider, manager, _ = native_provider
    provider.on_turn_start(1, '')
    provider.on_memory_write('add', 'user', 'The user prefers concise technical explanations.')
    assert provider._store.count(provider._namespace) == 1
    output = finish(manager)[0]
    assert 'Зеркало Markdown' in output
    assert 'Сохранений нет' not in output
