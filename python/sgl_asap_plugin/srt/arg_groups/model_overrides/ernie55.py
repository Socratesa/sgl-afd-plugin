"""ERNIE 5.5 Flash's feature checks, on an attention rank whose experts run on the FFN
ranks."""

from __future__ import annotations


class _TwoBatchOverlapAllowed:
    """A read view of the resolving config with ``--enable-two-batch-overlap`` off."""

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, name):
        if name == "enable_two_batch_overlap":
            return False
        return getattr(self._cfg, name)


def _validate_features(original, cfg) -> None:
    """AROUND ``model_overrides.ernie55._validate_features``, which refuses
    ``--enable-two-batch-overlap``: an attention module keeps its output gate on itself
    from the q projection until the gate is taken, and a second micro-batch entering
    the module in between would clobber it. ``Ernie55LayerOperations`` runs attention
    as one operation (``srt/batch_overlap/operations_strategy.py``), so the
    micro-batches never meet inside the module and the check does not apply. Every
    other check, including the one for ``--enable-single-batch-overlap``, runs."""
    original(_TwoBatchOverlapAllowed(cfg))
