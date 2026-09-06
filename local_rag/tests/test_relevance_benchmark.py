"""Harness checks; controlled vectors here are NOT Gemma quality evidence."""
from pathlib import Path

import pytest

from evaluation.relevance_benchmark import (
    DATA, ExistingClient, filtered, retrieve, select_threshold, verify_protocol,
)
from evaluation.relevance_baseline import BaselineStore
from local_rag.store import MemoryStore


def test_selection_protects_cross_language_recall():
    protocol = {'threshold_grid': [.3, .5, .7], 'maximum_recall_loss': .05}
    rows = [
        {'gold': ['a'], 'retrieved': ['a'], 'semantic': {'a': .8}, 'category': 'paraphrase', 'language': 'en'},
        {'gold': ['b'], 'retrieved': ['b'], 'semantic': {'b': .6}, 'category': 'cross_language', 'language': 'ru'},
        {'gold': [], 'retrieved': ['c'], 'semantic': {'c': .4}, 'category': 'near_noanswer', 'language': 'en'},
    ]
    result = select_threshold(rows, protocol)
    assert result['selected']['threshold'] == .5
    assert result['curve'][-1]['by_category']['cross_language']['recall_at_5'] == 0
    assert result['selected']['all']['noanswer_false_positive_rate'] == 0


def test_preregistered_splits_are_disjoint_and_balanced():
    import json
    protocol = verify_protocol()
    splits = [json.loads((DATA / f'{split}.json').read_text()) for split in protocol['splits']]
    assert not ({q['query'] for q in splits[0]['queries']} & {q['query'] for q in splits[1]['queries']})
    assert not ({d['text'] for d in splits[0]['documents']} & {d['text'] for d in splits[1]['documents']})
    for split in splits:
        assert len(split['documents']) == 12
        assert sum(bool(q['gold']) for q in split['queries']) == 24
        assert sum(not q['gold'] for q in split['queries']) == 16
        assert sum(q['category'] == 'cross_language' for q in split['queries']) == 12
        assert all(set(q['gold']) <= {d['id'] for d in split['documents']} for q in split['queries'])


def test_existing_client_never_falls_back_to_startup(monkeypatch):
    client = ExistingClient(Path('/unused'))
    calls = []

    def request(op, *args, **kwargs):
        calls.append(op)
        raise PermissionError('test unavailable service')

    monkeypatch.setattr(client, '_request', request)
    with pytest.raises(PermissionError):
        client.embed_document('Only synthetic text')
    assert calls == ['ping']


def test_calibration_filter_matches_actual_pre_top_k_admission(tmp_path):
    query = {'query': 'telescope warranty', 'gold': ['good'], 'language': 'en', 'category': 'paraphrase'}
    documents = [
        {'id': 'bad', 'text': 'telescope warranty unavailable', 'kind': 'durable', 'importance': 1., 'age_days': 0},
        {'id': 'good', 'text': 'Гарантия на два года', 'kind': 'episodic', 'importance': 0., 'age_days': 365},
    ]
    data = {'documents': documents, 'queries': [query]}
    vectors = {'documents': {'bad': [.3, .9539392014], 'good': [.8, .6]}, 'queries': {'telescope warranty': [1., 0.]}}
    before = retrieve(data, vectors, tmp_path / 'before.db', BaselineStore, limit=2)
    after = retrieve(data, vectors, tmp_path / 'after.db', MemoryStore, threshold=.5)
    assert before[0]['retrieved'] == ['bad', 'good']
    assert after[0]['retrieved'] == ['good']
    assert filtered(before, .5)[0]['retrieved'] == after[0]['retrieved']
