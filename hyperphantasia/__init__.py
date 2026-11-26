"""Hyperphantasia: a mental-model reasoning path for AI systems.

    >>> from hyperphantasia import reason
    >>> insight = reason("The cup is left of the plate. The fork is left of the plate. "
    ...                  "Is the fork left of the cup?")
    >>> insight.status.value
    'possible'
"""

from .attune import CalibrationLedger, ListenerModel, SelfModel
from .domains import Status
from .loop import Insight, PhantasiaLoop, reason
from .palace import MemoryPalace
from .scene import Scene
from .schema import scene_from_dict

__all__ = ["reason", "PhantasiaLoop", "Insight", "Status", "Scene", "scene_from_dict",
           "ListenerModel", "CalibrationLedger", "SelfModel", "MemoryPalace"]
__version__ = "0.1.0"
