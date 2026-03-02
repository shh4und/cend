"""
Skeleton generation using Dijkstra-based path finding.

This module implements the skeleton generation step of DF-Tracing, which
constructs a centerline by finding optimal paths from a seed point to
terminal points through the neuron volume.
"""

import heapq
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def generate_skeleton_from_seed(
    maximas_set: np.ndarray,
    seed_point: Tuple[int, int, int],
    pressure_field: np.ndarray,
    neuron_mask: np.ndarray,
    shape: Tuple[int, int, int],
    dataset_number: int = 0,
    pseudo_distance: Optional[np.ndarray] = None,
    tubular_directions: Optional[np.ndarray] = None,
    alpha_pressure: float = 1.0,
    alpha_distance: float = 0.3,
    alpha_direction: float = 0.2,
) -> np.ndarray:
    """
    Generates the skeleton using a single Dijkstra search from the seed point
    to all terminal points (maxima).

    This implements the path-finding step of DF-Tracing, where optimal paths
    are found through the centerline by favoring high pressure (distance from
    boundary) values.

    The cost function can optionally incorporate:
    - **pseudo_distance**: penalises steps that leave high-distance regions,
      keeping the path along the medial axis of the pseudo-distance map.
    - **tubular_directions**: penalises steps that deviate from the local
      tubular direction estimated from the Hessian of the distance field.

    Args:
        maximas_set: Coordinates of the terminal (maxima) points (N, 3).
        seed_point: The coordinates of the seed point (z, y, x).
        pressure_field: The pressure distance field.
        neuron_mask: The binary mask of the neuron region.
        shape: Shape of the volume (z, y, x).
        dataset_number: Dataset identifier for logging.
        pseudo_distance: Optional float field from grey-erosion (0–1). Higher
            values near the tube centre reduce cost.
        tubular_directions: Optional ``(Z,Y,X,3)`` unit-vector field giving the
            local tubular direction. Steps aligned with this direction are
            cheaper.
        alpha_pressure: Weight for the inverse-pressure cost term.
        alpha_distance: Weight for the inverse-pseudo-distance cost term.
        alpha_direction: Weight for the direction-deviation cost term.

    Returns:
        Array of coordinates (N, 3) representing the complete skeleton.
    """
    skeleton_set: Set[Tuple[int, int, int]] = {seed_point}
    delta = 1e-6  # Add delta to avoid division by zero

    use_pseudo = pseudo_distance is not None and alpha_distance > 0
    use_direction = tubular_directions is not None and alpha_direction > 0

    logger.info(
        f"OP_{dataset_number}: Starting skeleton generation with Dijkstra's search "
        f"(pressure={alpha_pressure:.2f}, distance={alpha_distance:.2f}, "
        f"direction={alpha_direction:.2f})."
    )

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
                # --- cost: inverse pressure (favour medial paths) ---
                cost_pressure = alpha_pressure / (pressure_field[neighbor] + delta)

                # --- cost: inverse pseudo-distance (stay on high-distance ridges) ---
                cost_distance = 0.0
                if use_pseudo:
                    cost_distance = alpha_distance / (pseudo_distance[neighbor] + delta)

                # --- cost: direction deviation (stay aligned with tube axis) ---
                cost_direction = 0.0
                if use_direction:
                    d_vec = tubular_directions[current_voxel]
                    if np.any(d_vec != 0):
                        step = np.array(neighbor, dtype=np.float64) - np.array(
                            current_voxel, dtype=np.float64
                        )
                        step_norm = np.linalg.norm(step)
                        if step_norm > 0:
                            step /= step_norm
                            # 1 - |cos(θ)|: 0 when aligned, 1 when perpendicular
                            alignment = 1.0 - abs(float(np.dot(step, d_vec)))
                            cost_direction = alpha_direction * alignment

                edge_weight = cost_pressure + cost_distance + cost_direction
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
