"""Methods implemented in this repository.

``TLSA``     -- training-free label space alignment (Stage 1),
``TLSA_ST``  -- self-training with the universal classifier (Stage 2).

Both subclass :class:`methods.base.BaseMethod`, which owns the parts that are
independent of the adaptation algorithm.  A new method only has to implement
``before_training`` / ``forward`` / ``predict`` and be registered below.
"""

from methods.base import BaseMethod
from methods.tlsa import TLSA
from methods.tlsa_st import TLSA_ST

method_classes = {
    'TLSA': TLSA,
    'TLSA_ST': TLSA_ST,
}

__all__ = ['BaseMethod', 'TLSA', 'TLSA_ST', 'method_classes']
