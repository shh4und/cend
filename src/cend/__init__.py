"""
CEND - Centerline Extraction using Neuron Direction

A Python package for automated 3D neuron reconstruction from microscopy images.
"""

__version__ = "0.1.0"

# Core classes and functions
from .core.distance_fields import DistanceFields
from .io.image import load_3d_volume, save_3d_volume
from .processing.multiscale import multiscale_filtering
from .structures.graph import Graph
from .structures.swc import SWCFile

__all__ = [
    "DistanceFields",
    "Graph",
    "SWCFile",
    "load_3d_volume",
    "save_3d_volume",
    "multiscale_filtering",
]
