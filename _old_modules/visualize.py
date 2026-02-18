import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def plot_projections(images, aggregations, axes, cmaps, title=None):
    """
    Plots 2D projections of 3D images in a grid with a maximum of 2 columns.

    Args:
        images (list or np.ndarray): A list of 3D numpy arrays (images) to be plotted.
        aggregations (list or str): A list of aggregation methods ('max', 'mean', 'min').
        axes (list or int): A list of axes (0, 1, or 2) for the projection.
        cmaps (list or str): A list of colormaps (e.g., 'viridis', 'gray').
    """
    # Ensure all inputs are lists for consistent iteration
    if not isinstance(images, list):
        images = [images]
    if not isinstance(aggregations, list):
        aggregations = [aggregations] * len(images)
    if not isinstance(axes, list):
        axes = [axes] * len(images)
    if not isinstance(cmaps, list):
        cmaps = [cmaps] * len(images)

    num_images = len(images)
    ncols = 2
    nrows = int(np.ceil(num_images / ncols))

    # Create the figure and subplots in a grid
    fig, axes_subplots = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

    # Flatten the axes array for easier iteration
    if num_images > 1:
        axes_flat = axes_subplots.flatten()
    else:
        axes_flat = [axes_subplots]

    for i in range(num_images):
        img = images[i]
        aggregation = aggregations[i]
        axis = axes[i]
        cmap = cmaps[i]

        # Perform aggregation
        if aggregation == "max":
            projected_img = np.max(img, axis=axis)
        elif aggregation == "mean":
            projected_img = np.mean(img, axis=axis)
        elif aggregation == "min":
            projected_img = np.min(img, axis=axis)
        else:
            raise ValueError(
                f"Unsupported aggregation method: {aggregation}. Choose from 'max', 'mean', 'min'."
            )

        set_title = (
            title + "Agg: " + str(aggregation) + " Axis: " + str(axis)
            if title is not None
            else "Agg: " + str(aggregation) + " Axis: " + str(axis)
        )
        # Plot the image in the correct subplot
        axes_flat[i].imshow(projected_img, cmap=cmap)
        axes_flat[i].set_title(f"{set_title}")
        axes_flat[i].axis("off")

    # Hide any unused subplots
    for j in range(num_images, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def show_vessel_tree_3d(vessel_tree, background_volume=None):
    """
    Creates an interactive 3D visualization of a vessel or neuron tree using Open3D.

    Each branch is represented as a colored line, with a sphere at its starting point.

    Args:
        vessel_tree (dict): A dictionary where keys are branch IDs and values are dicts
                            containing the 'centerline' (list of points) and 'parent' ID.
        background_volume (np.ndarray, optional): A 3D volume to display as a reference
                                                  isosurface in the background. Defaults to None.
    """
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="3D Vessel Tree Visualization", width=1280, height=800)

    # Generate a colormap for distinguishing different branches
    color_map = plt.cm.get_cmap("tab20", len(vessel_tree))

    for branch_id, branch_data in vessel_tree.items():
        centerline = branch_data["centerline"]
        if len(centerline) > 1:
            # Open3D expects (X, Y, Z) coordinates, while NumPy/Scikit-Image use (Z, Y, X).
            # We assume the input coordinates are (Z, Y, X) and swap them for visualization.
            points = np.array(centerline)[:, ::-1]  # Swap Z and X columns

            # Create a line set for the centerline
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            lines = [[i, i + 1] for i in range(len(points) - 1)]
            line_set.lines = o3d.utility.Vector2iVector(lines)

            # Assign a unique color to the branch
            color = color_map(branch_id % color_map.N)[:3]  # Get RGB from colormap
            line_set.paint_uniform_color(color)
            visualizer.add_geometry(line_set)

            # Add a sphere to mark the start of the branch
            start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            start_sphere.translate(points[0])
            start_sphere.paint_uniform_color(color)
            visualizer.add_geometry(start_sphere)

    # Add the background volume as a transparent isosurface if provided
    if background_volume is not None:
        add_background_mesh(visualizer, background_volume)

    # Configure rendering options
    setup_visualizer_options(visualizer)
    visualizer.run()
    visualizer.destroy_window()


def show_centerline_3d(centerline, background_volume=None):
    """
    Creates an interactive 3D visualization of a single centerline using Open3D.

    The centerline is shown in blue, with a green sphere at the start and a red one at the end.

    Args:
        centerline (list or np.ndarray): A list or array of points defining the centerline.
        background_volume (np.ndarray, optional): A 3D volume to display as a reference
                                                  isosurface. Defaults to None.
    """
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="3D Centerline Visualization", width=1280, height=800)

    if len(centerline) > 1:
        points = np.array(centerline)[:, ::-1]  # Swap Z,Y,X to X,Y,Z

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(points) - 1)]),
        )
        line_set.paint_uniform_color([0.0, 0.6, 1.0])  # Blue color
        visualizer.add_geometry(line_set)

        # Green sphere for the start point
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
        start_sphere.translate(points[0])
        start_sphere.paint_uniform_color([0.0, 1.0, 0.0])
        visualizer.add_geometry(start_sphere)

        # Red sphere for the end point
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
        end_sphere.translate(points[-1])
        end_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        visualizer.add_geometry(end_sphere)

    if background_volume is not None:
        add_background_mesh(visualizer, background_volume)

    setup_visualizer_options(visualizer)
    visualizer.run()
    visualizer.destroy_window()


def show_points_3d(
    points, background_volume=None, color=[1.0, 0.1, 0.1], point_size=2.0, threshold_percentile=80
):
    """
    Creates an interactive 3D visualization of a point cloud.

    Useful for displaying local maxima from VFC or other detected points.

    Args:
        points (np.ndarray): An (N, 3) array of (Z, Y, X) coordinates.
        background_volume (np.ndarray, optional): A 3D volume for context.
        color (list, optional): The [R, G, B] color for the points. Defaults to red.
        point_size (float, optional): The display size of the points. Defaults to 2.0.
    """
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="3D Point Cloud Visualization", width=1280, height=800)

    if points is not None and len(points) > 0:
        point_cloud = o3d.geometry.PointCloud()
        # Swap Z,Y,X to X,Y,Z for Open3D
        points_for_o3d = np.array(points)[:, ::-1]
        point_cloud.points = o3d.utility.Vector3dVector(points_for_o3d)
        point_cloud.paint_uniform_color(color)
        visualizer.add_geometry(point_cloud)
        logging.info(f"Visualizing {len(points)} points.")

    if background_volume is not None:
        add_background_mesh(visualizer, background_volume, threshold_percentile)

    setup_visualizer_options(
        visualizer,
        point_size=point_size,
    )
    visualizer.run()
    visualizer.destroy_window()


def show_volume_3d_dt(volume, distance_field, threshold_percentile=80, colormap=plt.cm.plasma):
    """
    Creates an interactive 3D isosurface visualization of a volume, with vertex
    colors mapped to the voxel intensities.

    Args:
        volume (np.ndarray): The 3D volume to visualize.
        threshold_percentile (int, optional): The percentile to use for the isosurface
                                              threshold. Defaults to 80.
        colormap (matplotlib.colors.Colormap, optional): The colormap for vertex colors.
                                                        Defaults to 'viridis'.
    """
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="3D Volume Visualization", width=1280, height=800)

    logging.info("Generating mesh from volume...")
    threshold = np.percentile(volume[volume > 0], threshold_percentile)

    # marching_cubes returns vertices in (Z, Y, X) order
    verts, faces, _, _ = marching_cubes(volume, level=threshold)

    # Create the mesh for the isosurface
    mesh = o3d.geometry.TriangleMesh()
    # Convert vertices to Open3D format (X, Y, Z) for correct orientation
    mesh.vertices = o3d.utility.Vector3dVector(verts[:, ::-1])
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # --- Apply Colormap Based on Voxel Intensity ---
    # 1. Get the intensity of the original volume at each vertex location.
    #    'map_coordinates' interpolates the values from the volume at the vertex coordinates.
    vert_distances = ndi.map_coordinates(distance_field, verts.T, order=1)

    # Etapa 5: Normalizar esses valores de distância para o intervalo [0, 1]
    min_dist, max_dist = vert_distances.min(), vert_distances.max()
    if (max_dist - min_dist) > 0:
        norm_distances = (vert_distances - min_dist) / (max_dist - min_dist)
    else:
        norm_distances = np.zeros_like(vert_distances)

    # 3. Apply the colormap to the normalized intensities.
    vertex_colors = colormap(norm_distances)[:, :3]  # Get RGB, ignore Alpha

    # 4. Assign the calculated colors to the mesh vertices.
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    visualizer.add_geometry(mesh)
    logging.info(
        f"Volume mesh added with threshold at percentile {threshold_percentile} and colormap."
    )

    setup_visualizer_options(visualizer)
    visualizer.run()
    visualizer.destroy_window()


def add_background_mesh(visualizer, volume, threshold_percentile=80):
    """Helper to add a background volume as a gray mesh to a visualizer."""
    logging.info("Generating background volume mesh...")
    threshold = np.percentile(volume[volume > 0], threshold_percentile)
    verts, faces, _, _ = marching_cubes(volume, level=threshold)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts[:, ::-1])  # Swap Z,Y,X to X,Y,Z
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
    mesh.compute_vertex_normals()

    visualizer.add_geometry(mesh)
    logging.info(f"Background mesh added with threshold at percentile {threshold_percentile}.")


def setup_visualizer_options(visualizer, point_size=3.0, line_width=2.0):
    """Helper to apply common rendering options to a visualizer."""
    opt = visualizer.get_render_option()
    opt.background_color = np.asarray([0.30, 0.30, 0.30])  # Dark background
    opt.point_size = point_size

    # Add a coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25.0, origin=[0, 0, 0])
    visualizer.add_geometry(coord_frame)

    # Set initial camera view
    visualizer.get_view_control().set_zoom(0.8)

    logging.info("Close the visualization window to continue script execution.")
