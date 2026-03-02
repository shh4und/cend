import argparse
import gc
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from tqdm import tqdm

from ..core.distance_fields import DistanceFields
from ..core.segmentation import adaptive_mean_mask
from ..core.skeletonization import generate_skeleton_from_seed
from ..core.utils import (
    compute_tubular_direction,
    create_maxima_image,
    find_pseudo_distance_maxima,
    strel_non_flat_sphere,
)
from ..io.image import load_3d_volume
from ..structures.graph import Graph
from .multiscale import multiscale_filtering, multiscale_on_distance


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
        grey_morpho_size,
        grey_morpho_weight,
    ) = args

    logging.info(f"Processing image {img_idx + 1}: {img_path.name}")

    # 1. Load image
    volume = load_3d_volume(str(img_path))

    # 2. Pre-processing: light Gaussian smoothing only (no grey morphology here)
    volume = ndi.gaussian_filter(volume, 1.0)

    # 3. Filtering: multi-scale tubularity on the smoothed raw volume
    img_filtered = multiscale_filtering(
        volume=volume,
        sigma_range=sigma_range,
        filter_type=filter_type,
        neuron_threshold=neuron_threshold,
        dataset_number=img_idx + 1,
    )

    # 4. Post-filtering morphology on float response (professor's key suggestion)
    #    Normalise to [0, 1] and apply non-flat erosion + closing on continuous data.
    #    The non-flat erosion with a parabolic SE approximates a distance transform:
    #    ridges of the resulting map align with tube centres.
    img_max = img_filtered.max()
    if img_max > 0:
        img_filtered /= img_max
    img_filtered = img_as_ubyte(img_filtered)
    struct_erosion = strel_non_flat_sphere(grey_morpho_weight, grey_morpho_size)
    img_filtered = ndi.grey_erosion(img_filtered, structure=struct_erosion)

    # Non-flat closing (preserves tubular geometry better than a flat cube)
    struct_closing = strel_non_flat_sphere(grey_morpho_weight, grey_morpho_size)
    img_filtered = ndi.grey_closing(img_filtered, structure=struct_closing)

    # Keep a copy of the continuous pseudo-distance map before quantising
    pseudo_distance = img_filtered.copy()

    # 5. Second-stage filtering: Hessian on the pseudo-distance map
    #    A narrower scale range picks up residual tubularity in the smoothed
    #    distance-like field, which is more robust in low-signal regions.
    img_filtered_2 = multiscale_on_distance(
        volume=img_filtered,
        sigma_range=(1, 4.75, 1.5),
        filter_type=filter_type,
        neuron_threshold=neuron_threshold,
        dataset_number=img_idx + 1,
    )

    # Combine both responses (max) — the first pass catches strong neurites,
    # the second pass recovers weak/thin ones via the distance-like field.
    combined = np.maximum(img_filtered, img_filtered_2)

    # 6. Segmentation: adaptive threshold on the combined response
    zero_t = filter_type != "yang"
    img_mask = adaptive_mean_mask(img_as_ubyte(combined), zero_t=zero_t)[0]
    del img_filtered_2, combined
    gc.collect()

    # 7. Distance fields
    df = DistanceFields(
        shape=volume.shape,
        seed_point=root_coord,
        dataset_number=img_idx + 1,
    )
    pressure_field = ndi.gaussian_filter(df.pressure_field(img_mask), 1.0)
    thrust_field = ndi.gaussian_filter(df.thrust_field(img_mask), 2.0)

    # 8. Seed finding: combine thrust maxima with pseudo-distance maxima
    #    Thrust maxima find endpoints; pseudo-distance maxima find medial-axis
    #    points that may be missed by thrust alone (reduces false seeds near
    #    irregular borders).
    thrust_seeds = df.find_thrust_maxima(thrust_field, img_mask, order=maximas_min_dist)
    pd_seeds = find_pseudo_distance_maxima(
        pseudo_distance, img_mask, order=maximas_min_dist, min_value=0.05
    )
    maximas_set = (
        np.unique(np.vstack([thrust_seeds, pd_seeds]), axis=0) if pd_seeds.size else thrust_seeds
    )

    # 9. Compute tubular directions from pseudo-distance Hessian for tracing guidance
    tubular_dirs = compute_tubular_direction(pseudo_distance, sigma=1.5)

    # 10. Skeletonization with enhanced Dijkstra cost
    skel_coords = generate_skeleton_from_seed(
        maximas_set,
        root_coord,
        pressure_field,
        img_mask,
        volume.shape,
        img_idx + 1,
        pseudo_distance=pseudo_distance,
        tubular_directions=tubular_dirs,
        alpha_pressure=1.0,
        alpha_distance=0.3,
        alpha_direction=0.7,
    )
    clean_skel = create_maxima_image(skel_coords, volume.shape)
    del img_mask, thrust_field, skel_coords, maximas_set, df, volume
    del pseudo_distance, tubular_dirs, img_filtered
    gc.collect()

    if not np.any(clean_skel):
        logging.error(f"Empty skeleton for image {img_idx + 1}. Skipping.")
        return None

    # Find the closest skeleton point to the original root coordinate
    skel_points = np.argwhere(clean_skel)
    distances_sq = np.sum((skel_points - np.array(root_coord)) ** 2, axis=1)
    initial_valid_root = tuple(skel_points[np.argmin(distances_sq)])

    g = Graph(clean_skel, initial_valid_root)
    del clean_skel
    gc.collect()

    g.calculate_mst()

    # 7. Pruning and root validation
    if not g.prune_and_reroot(pruning_threshold, root_coord):
        logging.error(f"Graph became empty after pruning for image {img_idx + 1}. Skipping.")
        return None

    # 8. Smooth the tree and save the SWC file
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
        _save_metadata(
            output_filename,
            img_path,
            filter_type,
            sigma_range,
            neuron_threshold,
            pruning_threshold,
            grey_morpho_size,
            grey_morpho_weight,
        )
        return str(output_filename)
    else:
        logging.error(f"Failed to save SWC file for image {img_idx + 1}.")
        return None


def _save_metadata(
    swc_path: Path,
    img_path: Path,
    filter_type: str,
    sigma_range: Tuple,
    neuron_threshold: float,
    pruning_threshold: int,
    grey_morpho_size: int,
    grey_morpho_weight: float,
) -> None:
    """Writes a .meta sidecar file alongside a generated SWC file."""
    meta_path = swc_path.with_suffix(".meta")
    try:
        with open(meta_path, "w") as f:
            f.write(f"source_image: {img_path.name}\n")
            f.write(f"filter_type: {filter_type}\n")
            f.write(f"sig_min: {sigma_range[0]}\n")
            f.write(f"sig_max: {sigma_range[1]}\n")
            f.write(f"sig_step: {sigma_range[2]}\n")
            f.write(f"neuron_threshold: {neuron_threshold}\n")
            f.write(f"pruning_threshold: {pruning_threshold}\n")
            f.write(f"grey_morpho_size: {grey_morpho_size}\n")
            f.write(f"grey_morpho_weight: {grey_morpho_weight}\n")
        logging.info(f"Metadata saved to {meta_path}")
    except Exception as e:
        logging.error(f"Failed to save metadata file: {e}")


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
        default=10,
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
        default=20,
        help="Number of points per branch for smoothing.",
    )
    parser.add_argument(
        "--grey_morpho_size",
        type=int,
        default=2,
        help="Size of grey morphological non-flat structure element",
    )
    parser.add_argument(
        "--grey_morpho_weight",
        type=float,
        default=0.5,
        help="Weight of grey morphological non-flat structure element",
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
                args.grey_morpho_size,
                args.grey_morpho_weight,
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
