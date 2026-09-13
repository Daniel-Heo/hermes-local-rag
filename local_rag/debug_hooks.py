"""Connect active memory providers to Hermes' public lifecycle dispatcher.

The memory discovery collector intentionally ignores register_hook. Register
through the real PluginContext after initialization instead. Registries hold
weak references: neither finished agents nor their memory can be kept alive by
process-wide hooks. No Hermes code or dispatcher is replaced.
"""
from __future__ import annotations

from pathlib import Path
import threading
import weakref

_LOCK = threading.RLock()
_REGISTRIES = weakref.WeakKeyDictionary()


def attach(provider) -> None:
    from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

    manager = get_plugin_manager()
    with _LOCK:
        providers = _REGISTRIES.get(manager)
        if providers is None:
            providers = weakref.WeakSet()
            _REGISTRIES[manager] = providers

            def matching(session_id):
                from hermes_constants import get_hermes_home
                home = Path(get_hermes_home()).resolve()
                with _LOCK:
                    current = list(providers)
                return [p for p in current if p._hermes_home == home
                        and p._session_id == session_id and p._service is not None]

            def transform(response_text, *, session_id='', **kwargs):
                candidates = matching(session_id)
                # Ambiguous ownership is not permission to reveal either trace.
                if len(candidates) != 1:
                    return None
                return candidates[0].transform_output(response_text, session_id=session_id)

            def clear(*, session_id='', **kwargs):
                for active in matching(session_id):
                    active.clear_debug_turn(session_id)

            context = PluginContext(PluginManifest(name='local_rag', kind='exclusive'), manager)
            context.register_hook('transform_llm_output', transform)
            context.register_hook('on_session_end', clear)
        providers.add(provider)
