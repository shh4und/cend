import logging
from collections import deque
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import splev, splprep

from .swc import SWCFile


class Graph:
    """
    A class to create, process, and save a graph from a 3D image skeleton.

    The main workflow is:
    1. Initialize with an image and a root voxel.
    2. Efficiently create the graph starting from the root.
    3. Compute the Minimum Spanning Tree (MST).
    4. Optionally, prune short branches from the MST.
    5. Label the MST nodes for the SWC format.
    6. Save the result to an .swc file, with optional smoothing.
    """

    def __init__(self, image: np.ndarray, root_voxel: Tuple[int, int, int]):
        """
        Initializes the Graph object.

        Args:
            image (np.ndarray): The 3D image (binary or grayscale).
            root_voxel (Tuple[int, int, int]): The (z, y, x) coordinates of the root node.
        """
        if not (
            0 <= root_voxel[0] < image.shape[0]
            and 0 <= root_voxel[1] < image.shape[1]
            and 0 <= root_voxel[2] < image.shape[2]
            and image[root_voxel] != 0
        ):
            raise ValueError("Root voxel is out of image bounds or has a zero value.")

        self.image = image
        self.shape = image.shape
        self.root = root_voxel

        self.graph = nx.Graph()
        self.mst: Optional[nx.Graph] = None

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("Starting graph creation from the root...")
        self._create_graph_from_root()

    def _create_graph_from_root(self) -> None:
        """
        Efficiently creates the graph using a Breadth-First Search (BFS) from the root.
        This avoids building a dense graph of the entire image, focusing only
        on the component connected to the root.
        """
        queue = deque([self.root])
        visited = {self.root}
        self.graph.add_node(self.root, pos=self.root)

        while queue:
            current_voxel = queue.popleft()

            # Find the 26 neighbors of the current voxel
            for neighbor in self._get_26_neighborhood(current_voxel):
                if neighbor not in visited:
                    visited.add(neighbor)

                    # Add the node and the edge with its Euclidean weight
                    self.graph.add_node(neighbor, pos=neighbor)
                    weight = self._euclidean_distance(current_voxel, neighbor)
                    self.graph.add_edge(current_voxel, neighbor, weight=weight)

                    queue.append(neighbor)

        self.logger.info(
            f"Graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges."
        )

    def calculate_mst(self) -> None:
        """Calculates the Minimum Spanning Tree (MST) and stores it in self.mst."""
        self.logger.info("Calculating the Minimum Spanning Tree (MST)...")
        if not self.graph:
            self.logger.warning("Graph is empty. Cannot calculate MST.")
            return

        self.mst = nx.minimum_spanning_tree(self.graph, weight="weight", algorithm="kruskal")
        self.logger.info("MST calculated.")

    def prune_mst_by_length(self, length_threshold: int):
        """
        Prunes short branches from the MST. Pruning is more efficient on the MST
        than on the full graph.

        Args:
            length_threshold (int): The maximum length (in number of nodes) for a branch to be pruned.
        """
        if not self.mst:
            self.logger.warning(
                "MST must be calculated before pruning. Call `calculate_mst()` first."
            )
            return
        if length_threshold <= 0:
            return

        self.logger.info(f"Pruning MST branches with length <= {length_threshold}")

        mst_copy = self.mst.copy()
        endpoints = [node for node, degree in mst_copy.degree() if degree == 1]
        nodes_to_remove = set()

        for start_node in endpoints:
            if start_node in nodes_to_remove:
                continue

            path = [start_node]
            current_node = start_node

            # Traverse the branch from a leaf until a junction (degree > 2) is found
            while mst_copy.degree(current_node) < 3:
                # In a tree, a degree-2 node has exactly one unvisited neighbor in the path
                neighbors = [n for n in mst_copy.neighbors(current_node) if n not in path]
                if not neighbors:
                    break  # Reached the end of an isolated fragment

                current_node = neighbors[0]
                path.append(current_node)

                # Safety break
                if len(path) > self.graph.number_of_nodes():
                    break

            # The last node in 'path' is the junction. The branch to prune doesn't include it.
            branch_to_prune = path[:-1]
            if len(branch_to_prune) <= length_threshold:
                nodes_to_remove.update(branch_to_prune)

        if nodes_to_remove:
            self.mst.remove_nodes_from(list(nodes_to_remove))
            self.logger.info(f"MST pruning complete. {len(nodes_to_remove)} nodes removed.")

    def validate_and_set_root(self, original_root: Tuple[int, int, int]) -> None:
        """
        Verifies if the original root is a node in the graph. If not, finds the
        closest node and updates self.root.

        Args:
            original_root (Tuple[int, int, int]): The coordinate of the original root.
        """
        if self.graph.has_node(original_root):
            self.root = original_root
            self.logger.info(f"Original root {original_root} is valid and present in the graph.")
            return

        self.logger.warning(
            f"Original root {original_root} not found in skeleton. Finding the closest node..."
        )

        nodes = np.array(list(self.graph.nodes()))
        # Calculate squared Euclidean distance from all nodes to the original root
        distances_sq = np.sum(
            (nodes - np.array(original_root)) ** 2, axis=1
        )  # BUG FIX: Added axis=1

        # Find the index of the node with the minimum distance
        closest_node_index = np.argmin(distances_sq)
        new_root = tuple(nodes[closest_node_index])

        self.root = new_root
        self.logger.info(f"Root updated to the closest node: {self.root}")

    def label_nodes_for_swc(self) -> None:
        """
        Performs a Depth-First Search (DFS) on the MST to label nodes with
        'id' and 'parent_id', preparing for the SWC format.
        """
        if not self.mst:
            self.logger.warning("MST must be calculated first. Call `calculate_mst()`.")
            return

        self.logger.info("Labeling nodes via DFS...")
        visited = set()
        stack = [(self.root, -1)]  # The stack contains (voxel, parent_id)
        node_id_counter = 1

        while stack:
            voxel, parent_id = stack.pop()
            if voxel not in visited:
                visited.add(voxel)

                # Ensure the node still exists in the MST after pruning
                if self.mst.has_node(voxel):
                    if "id" not in self.mst.nodes[voxel]:
                        self.mst.nodes[voxel]["id"] = node_id_counter
                        node_id_counter += 1

                    self.mst.nodes[voxel]["parent"] = parent_id

                    # Add neighbors to the stack
                    current_node_id = self.mst.nodes[voxel]["id"]
                    for neighbor in list(self.mst.neighbors(voxel)):
                        if neighbor not in visited:
                            stack.append((neighbor, current_node_id))

        self.logger.info("Node labeling complete.")

    def generate_smoothed_swc(
        self,
        filename: str,
        pressure_field: Optional[np.ndarray] = None,
        smoothing_factor: float = 0.5,
        num_points_per_branch: int = 20,
    ) -> bool:
        """
        Decomposes the MST into branches, smooths each with splines, and saves to an SWC file.

        Args:
            filename (str): The output filename (e.g., 'neuron_smoothed.swc').
            pressure_field (np.ndarray, optional): A 3D field with radius information.
            smoothing_factor (float): Spline smoothing parameter.
            num_points_per_branch (int): Number of points for each smoothed branch.

        Returns:
            bool: True if the file was saved successfully.
        """
        if not self.mst:
            self.logger.warning("MST must be calculated first.")
            return False

        self.logger.info("Starting spline smoothing and SWC generation...")
        swc = SWCFile(filename)

        # Map original voxels to new SWC IDs. Root is always ID 1 with parent -1.
        voxel_to_id_map = {self.root: 1}
        node_id_counter = 2  # Start at 2 since root is 1

        # Add the root point to the SWC
        z_root, y_root, x_root = self.root
        radius = 1.0
        if pressure_field is not None:
            radius = max(1.0, float(pressure_field[int(z_root), int(y_root), int(x_root)]))
        swc.add_point(1, 2, x_root, y_root, z_root, radius, -1)

        visited_edges = set()

        # Iterate over the tree from the root to maintain parent-child order
        for start_node in nx.dfs_preorder_nodes(self.mst, source=self.root):
            for neighbor in self.mst.neighbors(start_node):
                edge = tuple(sorted((start_node, neighbor)))
                if edge in visited_edges:
                    continue

                # 1. Find the complete branch
                path = [start_node, neighbor]
                curr = neighbor
                while self.mst.degree(curr) == 2:
                    next_node = [n for n in self.mst.neighbors(curr) if n != path[-2]][0]
                    path.append(next_node)
                    curr = next_node

                for i in range(len(path) - 1):
                    visited_edges.add(tuple(sorted((path[i], path[i + 1]))))

                # 2. Smooth the branch with a spline
                path_coords = np.array(path).T
                if len(path) < 4:
                    smoothed_points = np.array(path)
                else:
                    tck, u = splprep(path_coords, s=len(path) * smoothing_factor, k=3)
                    u_new = np.linspace(u.min(), u.max(), num_points_per_branch)
                    z_new, y_new, x_new = splev(u_new, tck)
                    smoothed_points = np.vstack([z_new, y_new, x_new]).T

                # 3. Add smoothed points to SWC with incrementing IDs
                parent_id = voxel_to_id_map[start_node]

                # Skip the first smoothed point as it corresponds to start_node
                for point_coords in smoothed_points[1:]:
                    current_id = node_id_counter
                    z, y, x = point_coords

                    radius = 1.0
                    if pressure_field is not None:
                        # Use map_coordinates for smooth radius interpolation
                        radius_val = ndi.map_coordinates(
                            pressure_field, [[z], [y], [x]], order=1, mode="nearest"
                        )[0]
                        radius = max(1.0, float(radius_val))

                    swc.add_point(current_id, 2, x, y, z, radius, parent_id)

                    parent_id = current_id
                    node_id_counter += 1

                # Update map with the ID of the last point of the branch
                voxel_to_id_map[path[-1]] = parent_id

        return swc.write_file()

    def save_to_swc(self, filename: str, pressure_field: Optional[np.ndarray] = None) -> bool:
        """
        Saves the labeled MST to an SWC format file.

        Args:
            filename (str): The output filename (e.g., 'neuron.swc').
            pressure_field (np.ndarray, optional): A 3D field with radius information.

        Returns:
            bool: True if the file was saved successfully.
        """
        if not self.mst:
            self.logger.warning("No MST to save.")
            return False

        swc = SWCFile(filename)

        # Sort nodes by ID for a well-formatted SWC file
        nodes_sorted = sorted(self.mst.nodes(data=True), key=lambda x: x[1].get("id", float("inf")))

        for node, attrs in nodes_sorted:
            if "id" not in attrs:
                continue

            z, y, x = map(float, node)
            radius = 1.0  # Default radius

            if pressure_field is not None:
                iz, iy, ix = int(z), int(y), int(x)
                if (
                    0 <= iz < pressure_field.shape[0]
                    and 0 <= iy < pressure_field.shape[1]
                    and 0 <= ix < pressure_field.shape[2]
                ):
                    radius = max(1.0, pressure_field[iz, iy, ix])

            swc.add_point(attrs["id"], 2, x, y, z, radius, attrs.get("parent", -1))

        self.logger.info(f"Saving MST to {filename}")
        return swc.write_file()

    # --- HELPER METHODS ---
    def _get_26_neighborhood(self, voxel: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Gets the 26-connected neighbors of a voxel that are part of the skeleton.
        """
        z, y, x = voxel
        neighbors = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue

                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (
                        0 <= nz < self.shape[0]
                        and 0 <= ny < self.shape[1]
                        and 0 <= nx < self.shape[2]
                        and self.image[nz, ny, nx] != 0
                    ):
                        neighbors.append((nz, ny, nx))
        return neighbors

    @staticmethod
    def _euclidean_distance(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
        """Calculates the Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))
