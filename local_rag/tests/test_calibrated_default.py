from local_rag.store import MemoryStore


def test_default_uses_calibration_selected_floor(tmp_path):
    store = MemoryStore(tmp_path / 'memory.db', dimensions=2)
    try:
        assert store.semantic_threshold == 0.3
    finally:
        store.close()
