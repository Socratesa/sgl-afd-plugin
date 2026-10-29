"""Exception types shared by the plugin modules."""


class AsapConfigError(ValueError):
    """The requested deployment or configuration is not supported (fixable by the user)."""


class AsapInstallError(RuntimeError):
    """Plugin hooks cannot be attached to this sglang build (environment problem)."""
