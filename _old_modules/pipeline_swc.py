import argparse
import gc
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from tqdm import tqdm

from dst_fields import DistanceFields
from graphs import Graph
from image_io import load_3d_volume
from vfc import create_maxima_image


def process_image(args: Tuple):
    """Executes the complete neuron reconstruction pipeline for a single image."""
    (
        img_idx,
        img_path,
        filter_type,
        root_coord,
        output_dir,
        sigma_range,
        neuron_threshold,
        pruning_threshold,
        maximas_min_dist,
        smoothing_factor,
        num_points_per_branch,
    ) = args

    logging.info(f"Processing image {img_idx + 1}: {img_path.name}")

    # 1. Load Image
    volume = load_3d_volume(str(img_path))

    # 2. Pre-processing
    gauss_filtered = ndi.gaussian_filter(volume, 1.0)
    min_filtered = ndi.minimum_filter(gauss_filtered, 2)
    volume[min_filtered == 0] = 0
    del gauss_filtered, min_filtered
    gc.collect()

    # 3. Filtering and Segmentation
    df = DistanceFields(
        volume=volume,
        filter_type=filter_type,
        sigma_range=sigma_range,
        neuron_threshold=neuron_threshold,
        seed_point=root_coord,
        dataset_number=img_idx + 1,
    )
    img_filtered = df.multiscale_filtering()
    img_grey_morpho = df.grey_morphological_denoising(img_filtered)
    del img_filtered
    gc.collect()
    img_mask = df.adaptive_mean_mask(
        img_grey_morpho, zero_t=True if filter_type != "yang" else False
    )[0]
    del img_grey_morpho
    gc.collect()

    # clean_img_mask = df.morphological_denoising(img_mask)
    pressure_field = ndi.gaussian_filter(df.pressure_field(img_mask), 2.0)
    thrust_field = ndi.gaussian_filter(df.thrust_field(img_mask), 1.0)
    # del img_mask
    gc.collect()

    # 4. Skeletonization
    maximas_set = df.find_thrust_maxima(thrust_field, img_mask, order=maximas_min_dist)
    skel_coords = df.generate_skel_from_seed(maximas_set, root_coord, pressure_field, img_mask)
    skel_img = create_maxima_image(skel_coords, volume.shape)
    clean_skel = skeletonize(skel_img)
    del img_mask, thrust_field, skel_img, skel_coords, maximas_set, df, volume
    gc.collect()

    if not np.any(clean_skel):
        logging.error(f"Empty skeleton for image {img_idx + 1}. Skipping.")
        return None

    # Find the closest point on the skeleton to the initial root coordinate
    skel_points = np.argwhere(clean_skel)
    distances_sq = np.sum((skel_points - np.array(root_coord)) ** 2, axis=1)
    initial_valid_root = tuple(skel_points[np.argmin(distances_sq)])

    g = Graph(clean_skel, initial_valid_root)
    del clean_skel
    gc.collect()

    g.calculate_mst()

    # 5. Pruning and Root Validation
    if pruning_threshold > 0:
        g.prune_mst_by_length(pruning_threshold)

        # If pruning removed the root, find a new one in the largest remaining component
        if not g.mst.has_node(g.root):
            logging.warning(f"Root {g.root} was removed during pruning. Finding a new root.")
            if g.mst.number_of_nodes() == 0:
                logging.error("Graph became empty after pruning. Skipping.")
                return None

            main_component = max(nx.connected_components(g.mst), key=len)

            # Find the node in the main component closest to the original root
            nodes_in_component = np.array(list(main_component))
            distances_to_original_root_sq = np.sum(
                (nodes_in_component - np.array(root_coord)) ** 2, axis=1
            )
            new_root = tuple(nodes_in_component[np.argmin(distances_to_original_root_sq)])

            g.root = new_root
            logging.info(f"New root set to {new_root}")

    # 6. Smooth the tree and save the SWC file
    output_filename = output_dir / f"OP_{img_idx + 1}_reconstruction.swc"
    success = g.generate_smoothed_swc(
        str(output_filename),
        pressure_field,
        smoothing_factor=smoothing_factor,
        num_points_per_branch=num_points_per_branch,
    )
    del pressure_field, g
    gc.collect()

    if success:
        logging.info(f"Image {img_idx + 1} successfully saved to {output_filename}")

        # Save a metadata file alongside the SWC
        meta_filename = output_filename.with_suffix(".meta")
        try:
            with open(meta_filename, "w") as meta_file:
                meta_file.write(f"source_image: {img_path.name}\n")
                meta_file.write(f"filter_type: {filter_type}\n")
                meta_file.write(f"sig_min: {sigma_range[0]}\n")
                meta_file.write(f"sig_max: {sigma_range[1]}\n")
                meta_file.write(f"sig_step: {sigma_range[2]}\n")
                meta_file.write(f"neuron_threshold: {neuron_threshold}\n")
                meta_file.write(f"pruning_threshold: {pruning_threshold}\n")
                meta_file.write(f"smoothing_factor: {smoothing_factor}\n")
                meta_file.write(f"num_points_per_branch: {num_points_per_branch}\n")
            logging.info(f"Metadata saved to {meta_filename}")
        except Exception as e:
            logging.error(f"Failed to save metadata file: {e}")

        return str(output_filename)
    else:
        logging.error(f"Failed to save SWC file for image {img_idx + 1}.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline for neuron reconstruction and SWC file generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the image folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_swc",
        help="Directory to save the SWC files.",
    )
    parser.add_argument(
        "--image_index",
        type=int,
        default=None,
        help="Index of the image to process (1-based). If not set, processes all.",
    )
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=2,
        help="Number of parallel processes. Default: 2 for memory safety.",
    )
    parser.add_argument(
        "--filter_type",
        type=str,
        default="yang",
        help="Choose which tubular filter to apply.",
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=1.0,
        help="Minimum sigma for multi-scale filtering.",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=2.0,
        help="Maximum sigma for multi-scale filtering.",
    )
    parser.add_argument(
        "--sigma_step",
        type=float,
        default=0.5,
        help="Sigma step for multi-scale filtering.",
    )
    parser.add_argument(
        "--neuron_threshold", type=float, default=0.05, help="Tubularity threshold."
    )
    parser.add_argument(
        "--pruning_threshold",
        type=int,
        default=0,
        help="Maximum length (in nodes) of a branch to be pruned. Set to 0 to disable.",
    )
    parser.add_argument(
        "--maximas_min_dist",
        type=int,
        default=2,
        help="Window size for finding local maxima.",
    )
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.8,
        help="Smoothing factor for spline interpolation.",
    )
    parser.add_argument(
        "--num_points_per_branch",
        type=int,
        default=15,
        help="Number of points per branch for smoothing.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in data_dir.glob("OP_*") if p.is_dir()])

    # Dataset-specific root coordinates
    roots = [
        (0, 429, 31),
        (25, 391, 1),
        (38, 179, 94),
        (0, 504, 128),
        (33, 264, 185),
        (10, 412, 15),
        (39, 216, 120),
        (55, 181, 119),
        (4, 364, 64),
    ]

    if not image_paths or len(image_paths) != len(roots):
        logging.error("Image directories not found or number of roots does not match.")
        return

    sigma_range = (args.sigma_min, args.sigma_max, args.sigma_step)

    tasks = []
    task_indices = range(len(image_paths))
    if args.image_index is not None:
        if not (1 <= args.image_index <= len(image_paths)):
            logging.error(
                f"Invalid image index: {args.image_index}. Must be between 1 and {len(image_paths)}."
            )
            return
        task_indices = [args.image_index - 1]

    for i in task_indices:
        tasks.append(
            (
                i,
                image_paths[i],
                args.filter_type,
                roots[i],
                output_dir,
                sigma_range,
                args.neuron_threshold,
                args.pruning_threshold,
                args.maximas_min_dist,
                args.smoothing_factor,
                args.num_points_per_branch,
            )
        )

    if not tasks:
        logging.warning("No tasks to execute.")
        return

    num_jobs = min(args.parallel_jobs, len(tasks), cpu_count())

    if num_jobs <= 1:
        logging.info(f"Executing {len(tasks)} task(s) sequentially.")
        for task in tqdm(tasks, desc="Processing Images"):
            process_image(task)
    else:
        logging.info(f"Executing {len(tasks)} task(s) in parallel with {num_jobs} jobs.")
        with Pool(processes=num_jobs) as pool:
            list(
                tqdm(
                    pool.imap_unordered(process_image, tasks),
                    total=len(tasks),
                    desc="Processing Images",
                )
            )

    logging.info("Pipeline finished.")


if __name__ == "__main__":
    main()
