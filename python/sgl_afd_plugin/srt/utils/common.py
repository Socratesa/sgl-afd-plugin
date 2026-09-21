from __future__ import annotations

import contextlib
import importlib


@contextlib.contextmanager
def override_mod(path: str, value):
    """Temporarily rebind one "module.attr" path; restore on exit."""
    mod_path, name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    old = getattr(mod, name)
    setattr(mod, name, value)
    try:
        yield
    finally:
        setattr(mod, name, old)


@contextlib.contextmanager
def override_obj(obj: object, name: str, value):
    """Temporarily rebind one attribute on an object; restore on exit."""
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def override_list(container, keys, value):
    """Temporarily set container[k] = value for k in keys; restore on exit."""
    saved = [(k, container[k]) for k in keys]
    for k, _ in saved:
        container[k] = value
    try:
        yield
    finally:
        for k, old in saved:
            container[k] = old
