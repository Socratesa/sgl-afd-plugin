from .expert_location import install as _install_expert_location


def install():
    """Install all hooks."""
    _install_expert_location()


__all__ = ("install",)
