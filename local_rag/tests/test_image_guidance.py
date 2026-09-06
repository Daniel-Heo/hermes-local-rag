"""Image workflow contracts; controlled vectors do not measure visual quality."""
import json

import pytest
from PIL import Image

from local_rag import LocalRagProvider
from local_rag.config import LocalRagConfig
from local_rag.tests.test_curated_images import Text, Clip


def assert_recall_guidance(value):
    for term in ('candidate', 'evidence', 'vision_analyze', 'pixels', 'exact colors', 'table cell', 'abstain'):
        assert term in value.lower()


def test_image_save_schema_guides_prior_pixel_inspection_without_attestation():
    schemas = {s['name']: s for s in LocalRagProvider(embedder=Text()).get_tool_schemas()}
    save = schemas['local_rag_index_image']
    props = save['parameters']['properties']
    assert 'description' not in save['parameters']['required']
    assert set(props) == {'path', 'decision', 'reason', 'scope', 'group', 'description'}
    for term in ('prior vision_analyze', 'pixels', 'colors', 'shapes', 'relationships', 'uncertain'):
        assert term in props['description']['description'].lower()
    assert 'not automatically' in save['description'].lower()
    for name in ('local_rag_search', 'local_rag_search_images'):
        assert_recall_guidance(schemas[name]['description'])
    assert_recall_guidance(LocalRagProvider(embedder=Text()).system_prompt_block())


@pytest.mark.parametrize('description', [None, 'A red rectangle on a plain background.'])
def test_native_provider_preserves_optional_caption_and_scoped_candidate_recall(tmp_path, description):
    root = tmp_path / 'project'
    root.mkdir()
    source = root / 'image.png'
    Image.new('RGB', (32, 24), 'red').save(source)
    home = tmp_path / 'home'
    LocalRagConfig(visual_enabled=True).save(home)
    p = LocalRagProvider(embedder=Text(), visual_embedder=Clip())
    p.initialize('s1', hermes_home=str(home), cwd=str(root))
    p._images.ocr = lambda _: ('fixture label', 'fixture')
    try:
        args = dict(path=str(source), decision='save', reason='Retain design reference', scope='session')
        if description is not None:
            args['description'] = description
        saved = json.loads(p.handle_tool_call('local_rag_index_image', args))
        assert saved['description'] == (description or '')
        assert 'verified' not in saved
        for name in ('local_rag_search', 'local_rag_search_images'):
            results = json.loads(p.handle_tool_call(name, {'query': 'fixture label'}))['results']
            assert results[0]['id'] == saved['id']
            assert results[0]['description'] == (description or '')
        assert_recall_guidance(p.prefetch('fixture label'))
        p.on_session_switch('s2')
        assert not json.loads(p.handle_tool_call('local_rag_search_images', {'query': 'fixture label'}))['results']
    finally:
        p.shutdown()
