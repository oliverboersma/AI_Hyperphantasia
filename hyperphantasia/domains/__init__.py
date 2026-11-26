"""Imaginative domains: each one is a different kind of mental model."""

from .base import DECISIVE, ChannelView, Domain, Imagination, Status, Verdict
from .categorical import CategoricalDomain
from .process import ProcessDomain
from .spatial import SpatialDomain

DOMAINS: dict[str, Domain] = {
    "spatial": SpatialDomain(),
    "categorical": CategoricalDomain(),
    "process": ProcessDomain(),
}

__all__ = ["DOMAINS", "DECISIVE", "ChannelView", "Domain", "Imagination", "Status", "Verdict",
           "SpatialDomain", "CategoricalDomain", "ProcessDomain"]
