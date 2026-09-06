#!/usr/bin/env python3
"""Text relevance calibration with an existing Gemma service and temporary DBs.

No provider.initialize(), production database access, service startup, or OCR.
Both splits and the selection rule are hashed before any inference. Heldout is
evaluated only after selection.json is persisted. No runtime files are edited.
"""
from __future__ import annotations

import argparse
from array import array
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_rag.inference import InferenceClient
from local_rag.provider import LocalRagProvider
from local_rag.service import LocalRagService
from local_rag.store import MemoryStore
from evaluation.relevance_baseline import BaselineStore

DATA = ROOT / 'evaluation' / 'relevance'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class ExistingClient(InferenceClient):
    def ensure_started(self):
        self._request('ping')  # Never call the lifecycle/startup implementation.


class CachedQueries:
    def __init__(self, vectors, dimensions):
        self.vectors, self.dimensions = vectors, dimensions

    def embed_query(self, text):
        return self.vectors[text]


def metrics(rows):
    yes = [r for r in rows if r['gold']]
    no = [r for r in rows if not r['gold']]

    def recall(k):
        return sum(len(set(r['retrieved'][:k]) & set(r['gold'])) / len(r['gold']) for r in yes) / len(yes) if yes else None

    ranks = [next((i for i, key in enumerate(r['retrieved'][:5], 1) if key in r['gold']), 0) for r in yes]
    return {
        'answerable': len(yes), 'noanswer': len(no),
        'recall_at_3': recall(3), 'recall_at_5': recall(5),
        'mrr_at_5': sum(1 / rank if rank else 0 for rank in ranks) / len(yes) if yes else None,
        'noanswer_false_positive_rate': sum(bool(r['retrieved']) for r in no) / len(no) if no else None,
    }


def breakdown(rows):
    return {
        'all': metrics(rows),
        'by_category': {c: metrics([r for r in rows if r['category'] == c]) for c in sorted({r['category'] for r in rows})},
        'by_language': {lang: metrics([r for r in rows if r['language'] == lang]) for lang in ('en', 'ru')},
    }


def verify_protocol():
    protocol = json.loads((DATA / 'protocol.json').read_text())
    for split, expected in protocol['splits'].items():
        if digest(DATA / f'{split}.json') != expected:
            raise ValueError(f'{split} changed since preregistration')
    return protocol


def embed_split(client, split, run):
    data = json.loads((DATA / f'{split}.json').read_text())
    documents, queries = {}, {}
    for doc in data['documents']:
        documents[doc['id']] = client.embed_document(doc['text'])
    for query in data['queries']:
        queries[query['query']] = client.embed_query(query['query'])
    vectors = {'documents': documents, 'queries': queries}
    if any(len(v) != client.dimensions for group in vectors.values() for v in group.values()):
        raise ValueError('Shared service returned an unexpected embedding dimension')
    if any(not all(math.isfinite(x) for x in v) for group in vectors.values() for v in group.values()):
        raise ValueError('Shared service returned non-finite embeddings')
    dump(run / f'{split}-embeddings.json', vectors)
    return data, vectors


def retrieve(data, vectors, database, store_type, *, threshold=None, limit=5):
    options = {} if threshold is None else {'semantic_threshold': threshold}
    dimensions = len(next(iter(vectors['documents'].values())))
    with closing(store_type(database, dimensions=dimensions, **options)) as store:
        now = time.time()
        for doc in data['documents']:
            store.add('synthetic', doc['text'], vectors['documents'][doc['id']],
                      source=doc['id'], kind=doc['kind'], importance=doc['importance'])
            store._conn.execute('UPDATE memories SET created_at=? WHERE source=?',
                                (now - doc['age_days'] * 86400, doc['id']))
        store._conn.commit()
        embedder = CachedQueries(vectors['queries'], dimensions)
        provider = LocalRagProvider(embedder=embedder)
        provider._service = LocalRagService(store=store, embedder=embedder, namespace='synthetic')
        provider._write_enabled = False
        # No initialize: the only database handle is the temporary store above.
        rows = []
        for q in data['queries']:
            results = provider._unified_search(q['query'], limit=limit)
            if provider._retrieval_warnings:
                raise RuntimeError(provider._retrieval_warnings)
            semantic = {key: MemoryStore._cosine(vectors['queries'][q['query']], array('f', value))
                        for key, value in vectors['documents'].items()}
            rows.append({**q, 'retrieved': [r['source'] for r in results],
                         'scores': [r['score'] for r in results], 'semantic': semantic})
        return rows


def filtered(rows, threshold):
    return [{**r, 'retrieved': [key for key in r['retrieved'] if r['semantic'][key] >= threshold][:5]}
            for r in rows]


def select_threshold(rows, protocol):
    # Only exhaustive CALIBRATION baseline ranks enter this function. Filtering
    # before truncation reproduces unchanged ranking among surviving candidates.
    baseline = breakdown(rows)
    curve, feasible = [], []
    for threshold in protocol['threshold_grid']:
        report = breakdown(filtered(rows, threshold))
        candidate = {'threshold': threshold, **report}
        curve.append(candidate)
        loss = protocol['maximum_recall_loss']
        if (report['all']['recall_at_5'] >= baseline['all']['recall_at_5'] - loss - 1e-12
                and report['by_category']['cross_language']['recall_at_5'] >=
                baseline['by_category']['cross_language']['recall_at_5'] - loss - 1e-12):
            feasible.append(candidate)
    selected = min(feasible, key=lambda c: (c['all']['noanswer_false_positive_rate'],
                   -c['all']['recall_at_5'], c['threshold'])) if feasible else None
    return {'baseline': baseline, 'curve': curve, 'selected': selected}


def calibrate(client, run, protocol):
    if (run / 'selection.json').exists():
        raise ValueError('Selection is already frozen; use a new run for a new experiment')
    data, vectors = embed_split(client, 'calibration', run)
    with tempfile.TemporaryDirectory(prefix='db-', dir=run) as directory:
        root = Path(directory)
        baseline = retrieve(data, vectors, root / 'before.db', BaselineStore, limit=len(data['documents']))
        dump(run / 'calibration-before.json', baseline)
        selection = select_threshold(baseline, protocol)
        selection.update(protocol_sha256=digest(DATA / 'protocol.json'),
                         baseline_sha256=digest(ROOT / 'evaluation/relevance_baseline.py'),
                         implementation_sha256=digest(ROOT / 'local_rag/store.py'))
        if selection['selected'] is not None:
            threshold = selection['selected']['threshold']
            actual = retrieve(data, vectors, root / 'after.db', MemoryStore, threshold=threshold)
            if [r['retrieved'] for r in actual] != [r['retrieved'] for r in filtered(baseline, threshold)]:
                raise AssertionError('Calibration simulation differs from actual production retrieval')
            dump(run / 'calibration-after.json', actual)
        dump(run / 'selection.json', selection)  # BEFORE loading heldout vectors/labels.
        if selection['selected'] is None:
            raise RuntimeError('No calibration threshold meets the preregistered recall constraint')
    return selection


def heldout(client, run):
    if (run / 'heldout-report.json').exists():
        raise ValueError('Heldout was already evaluated for this frozen selection')
    selection_path = run / 'selection.json'
    selection_hash = digest(selection_path)
    selection = json.loads(selection_path.read_text())
    if (selection['protocol_sha256'] != digest(DATA / 'protocol.json')
            or selection['implementation_sha256'] != digest(ROOT / 'local_rag/store.py')
            or selection['baseline_sha256'] != digest(ROOT / 'evaluation/relevance_baseline.py')):
        raise ValueError('Code or protocol changed since calibration; do not retune on heldout')
    threshold = selection['selected']['threshold']
    data, vectors = embed_split(client, 'heldout', run)
    outputs = {}
    with tempfile.TemporaryDirectory(prefix='db-', dir=run) as directory:
        for label, cls, value in [('before', BaselineStore, None),
                                  ('after_selected', MemoryStore, threshold),
                                  ('after_repository_default', MemoryStore, None)]:
            outputs[label] = retrieve(data, vectors, Path(directory) / f'{label}.db', cls, threshold=value)
            dump(run / f'heldout-{label}.json', outputs[label])
    report = {'selection_sha256': selection_hash, 'selected_threshold': threshold,
              'metrics': {label: breakdown(rows) for label, rows in outputs.items()},
              'recall_losses': {}, 'recall_gains': {}}
    for label in ('after_selected', 'after_repository_default'):
        pairs = list(zip(outputs['before'], outputs[label]))
        hit = lambda r: bool(set(r['gold']) & set(r['retrieved']))
        report['recall_losses'][label] = [{'before': b, 'after': a} for b, a in pairs if hit(b) and not hit(a)]
        report['recall_gains'][label] = [{'before': b, 'after': a} for b, a in pairs if not hit(b) and hit(a)]
    if digest(selection_path) != selection_hash:
        raise ValueError('Selection changed during heldout evaluation')
    dump(run / 'heldout-report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['calibrate', 'heldout', 'run'])
    parser.add_argument('--run', type=Path, required=True, help='New artifact directory under evaluation/')
    parser.add_argument('--service-home', type=Path, default=Path.home() / '.hermes',
                        help='Home of the existing inference service (default: ~/.hermes)')
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((ROOT / 'evaluation').resolve()):
        parser.error('--run must be inside this repository evaluation directory')
    run.mkdir(parents=True, exist_ok=True)
    protocol = verify_protocol()
    client = ExistingClient(args.service_home.expanduser().resolve(), dimensions=protocol['dimensions'], timeout=120)
    status = {'phase': args.phase, 'status': 'running', 'protocol_sha256': digest(DATA / 'protocol.json'),
              'dimensions': client.dimensions, 'python': sys.executable,
              'production_database_access': False, 'service_start_allowed': False}
    status_path = run / f'{args.phase}-status.json'
    if status_path.exists():
        parser.error('Phase already attempted; preserve artifacts and use a new run directory')
    dump(status_path, status)
    try:
        status['service_before'] = client._request('ping')
        if args.phase in ('calibrate', 'run'):
            calibrate(client, run, protocol)
        if args.phase in ('heldout', 'run'):
            heldout(client, run)
        status['service_after'] = client._request('ping')
        status['status'] = 'complete'
    except Exception as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        dump(status_path, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
