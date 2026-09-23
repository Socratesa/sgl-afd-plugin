from .serving_hook import install as _install_serving_hook
from .validation_hook import install as _install_validation_hook
from .parallel_hook import install as _install_parallel_hook


def install():
    """Install all hooks."""
    _install_serving_hook()
    _install_validation_hook()
    _install_parallel_hook()

__all__ = ("install",)
