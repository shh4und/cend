"""
Skeleton generation using Dijkstra-based path finding.

This module implements the skeleton generation step of DF-Tracing, which
constructs a centerline by finding optimal paths from a seed point to
terminal points through the neuron volume.
"""

import heapq
import logging
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def generate_skeleton_from_seed(
    maximas_set: np.ndarray,
    seed_point: Tuple[int, int, int],
    pressure_field: np.ndarray,
    neuron_mask: np.ndarray,
    shape: Tuple[int, int, int],
    dataset_number: int = 0,
) -> np.ndarray:
    """
    Generates the skeleton using a single Dijkstra search from the seed point
    to all terminal points (maxima).

    This implements the path-finding step of DF-Tracing, where optimal paths
    are found through the centerline by favoring high pressure (distance from
    boundary) values.

    Args:
        maximas_set: Coordinates of the terminal (maxima) points (N, 3).
        seed_point: The coordinates of the seed point (z, y, x).
        pressure_field: The pressure distance field.
        neuron_mask: The binary mask of the neuron region.
        shape: Shape of the volume (z, y, x).
        dataset_number: Dataset identifier for logging.

    Returns:
        Array of coordinates (N, 3) representing the complete skeleton.
    """
    skeleton_set: Set[Tuple[int, int, int]] = {seed_point}
    delta = 1e-6  # Add delta to avoid division by zero

    logger.info(f"OP_{dataset_number}: Starting skeleton generation with Dijkstra's search.")

    # Data structures for Dijkstra's algorithm from the seed
    distances: Dict[Tuple[int, int, int], float] = {seed_point: 0}
    previous_nodes: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    pq: List[Tuple[float, Tuple[int, int, int]]] = [(0, seed_point)]

    while pq:
        current_cost, current_voxel = heapq.heappop(pq)

        if current_cost > distances.get(current_voxel, float("inf")):
            continue

        # Explore neighbors
        for neighbor in _get_26_neighborhood(current_voxel, shape):
            if neuron_mask[neighbor]:
                # Cost is the inverse of pressure to favor paths through the neurite center
                edge_weight = 1.0 / (pressure_field[neighbor] + delta)
                new_cost = current_cost + edge_weight

                if new_cost < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_cost
                    previous_nodes[neighbor] = current_voxel
                    heapq.heappush(pq, (new_cost, neighbor))

    logger.info(f"OP_{dataset_number}: Dijkstra's search complete. Reconstructing paths...")

    # Reconstruct the path from each maximum back to the seed
    for maxima_point in maximas_set:
        current = tuple(maxima_point.astype(int).tolist())

        if current not in distances:
            # Skip maxima that were not reached by the search
            continue

        # Trace the path from the maximum back to the seed using predecessors
        while current in previous_nodes:
            skeleton_set.add(current)
            current = previous_nodes.get(current)

    logger.info(f"OP_{dataset_number}: Branch reconstruction complete.")

    if not skeleton_set:
        return np.array([], dtype=int).reshape(0, 3)

    return np.array(list(skeleton_set), dtype=int)


def _get_26_neighborhood(
    voxel: Tuple[int, int, int], shape: Tuple[int, int, int]
) -> List[Tuple[int, int, int]]:
    """
    Gets the 26-connected neighbors of a voxel within volume bounds.

    Args:
        voxel: The center voxel (z, y, x).
        shape: Shape of the volume (z, y, x).

    Returns:
        List of valid neighbor coordinates.
    """
    z, y, x = voxel
    neighbors = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                    neighbors.append((nz, ny, nx))
    return neighbors
