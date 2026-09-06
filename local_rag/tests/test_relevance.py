"""Admission regressions use controlled cosines, not model-quality estimates."""
import math

import pytest

from local_rag.provider import LocalRagProvider
from local_rag.service import LocalRagService
from local_rag.store import MemoryStore


def vector(cosine):
    return [cosine, math.sqrt(1 - cosine * cosine)]


@pytest.fixture
def store(tmp_path):
    value = MemoryStore(tmp_path / 'memory.db', dimensions=2, semantic_threshold=.5)
    yield value
    value.close()


@pytest.mark.parametrize('text,cosine,kind,importance', [
    ('The telescope was repaired yesterday.', .30, 'episodic', .5),
    ('Telescope lens catalog', .30, 'durable', 1.0),
    ('An unrelated durable preference', .40, 'durable', 1.0),
    ('The telescope was repaired yesterday.', 0.0, 'durable', 1.0),
])
def test_lexical_and_metadata_bonuses_cannot_admit_weak_match(store, text, cosine, kind, importance):
    store.add('owner', text, vector(cosine), source='synthetic', kind=kind, importance=importance)
    assert store.search('owner', 'What is the telescope warranty?', [1., 0.]) == []
    assert store._conn.execute('SELECT access_count FROM memories').fetchone()[0] == 0


@pytest.mark.parametrize('query,text', [
    ('How often are backups removed?', 'Резервные копии удаляются еженедельно.'),
    ('Когда удаляются резервные копии?', 'Backups are discarded every week.'),
])
def test_semantic_paraphrases_need_no_shared_words(store, query, text):
    store.add('owner', text, vector(.75), source='synthetic', importance=0)
    store._conn.execute('UPDATE memories SET created_at=0')
    assert [r.text for r in store.search('owner', query, [1., 0.])] == [text]


def test_gate_runs_before_top_k_and_access_updates(store):
    store.add('owner', 'The telescope warranty details are unavailable', vector(.40),
              source='weak', kind='durable', importance=1)
    store.add('owner', 'Гарантийный срок составляет два года', vector(.75), source='relevant', importance=0)
    # The weak row's old hybrid score exceeds the strong semantic-only score.
    assert [r.source for r in store.search('owner', 'What is the telescope warranty?', [1., 0.], limit=1)] == ['relevant']
    assert dict(store._conn.execute('SELECT source,access_count FROM memories')) == {'weak': 0, 'relevant': 1}


def test_weak_text_cannot_take_an_image_slot_in_rrf(store):
    class Embedder:
        dimensions = 2

        def embed_query(self, query):
            return [1., 0.]

    store.add('owner', 'The telescope catalog', vector(.3), source='weak', kind='durable', importance=1)
    provider = LocalRagProvider(embedder=Embedder())
    provider._service = LocalRagService(store=store, embedder=Embedder(), namespace='owner')
    image = {'id': 7, 'kind': 'image', 'score': .3, 'source': 'image:7', 'text': 'synthetic image'}
    provider._image_search = lambda query, limit: [image]
    assert provider._unified_search('What is the telescope warranty?', limit=1) == [image]


@pytest.mark.parametrize('embedding', [[0., 0.], [float('nan'), 1.], [float('inf'), 1.]])
def test_invalid_semantic_evidence_cannot_use_exact_lexical_match(store, embedding):
    store.add('owner', 'telescope warranty', embedding, source='invalid', kind='durable', importance=1)
    assert store.search('owner', 'telescope warranty', [1., 0.]) == []
