# CEND - Centerline Extraction using Neuron Direction

A Python implementation for automated 3D neuron reconstruction from microscopy images using distance field-based tracing and vector field convolution techniques.

## Overview

CEND implements a robust pipeline for extracting neuron centerlines from 3D microscopy volumes. The method combines multi-scale tubular filtering, distance field computation, and graph-based skeletonization to automatically trace neural structures and export them in the standard SWC format.

The pipeline is designed for processing neuron datasets and provides:

- **Multi-scale tubular filtering** for neuron enhancement
- **Distance field-based tracing** for accurate centerline extraction
- **Graph-based skeletonization** with minimum spanning tree computation
- **Automated pruning** to remove spurious branches
- **SWC format export** compatible with neuroscience analysis tools

## Features

- **Flexible Filtering**: Supports Yang tubularity filtering with configurable sigma ranges
- **Adaptive Segmentation**: Automatic thresholding with morphological denoising
- **Vector Field Computation**: Pressure and thrust field generation for directional tracing
- **Smart Skeletonization**: Seed-based skeleton generation with local maxima detection
- **Graph Processing**: MST computation with optional branch pruning
- **Batch Processing**: Parallel processing of multiple image volumes
- **Quality Evaluation**: Integrated DiademMetric scoring against gold standards
- **Visualization Tools**: 2D projections and 3D rendering utilities

## Installation

### Prerequisites

- Python 3.8+
- Java Runtime Environment (for DiademMetric evaluation)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/cend.git
cd cend

# Install in development mode
pip install -e .

# Or with optional dependencies
pip install -e ".[dev,notebooks]"
```

### Dependencies

Core dependencies (installed automatically):

- numpy, scipy, scikit-image
- networkx, opencv-python
- matplotlib, open3d
- tqdm

### DiademMetric Setup

For evaluation purposes, ensure the DiademMetric JAR file is present:

```bash
./metrics/DiademMetric/DiademMetric.jar
```

## Usage

### Command Line Interface

Process all images in the dataset:

```bash
cend --data_dir ./data --output_dir ./results_swc
```

Process a single image by index (1-based):

```bash
cend --image_index 1
```

### Python API

Use CEND programmatically in your scripts:

```python
from cend import load_3d_volume, multiscale_filtering, DistanceFields, Graph
from cend.core.segmentation import adaptive_mean_mask, grey_morphological_denoising
from cend.core.skeletonization import generate_skeleton_from_seed
from cend.core.vector_fields import create_maxima_image
import numpy as np
from scipy import ndimage as ndi

# Load data
volume = load_3d_volume("./data/OP_1")

# Apply multi-scale filtering
filtered = multiscale_filtering(
    volume,
    sigma_range=(1.0, 2.0, 0.5),
    filter_type="yang",
    neuron_threshold=0.05
)

# Segment and extract distance fields
mask = adaptive_mean_mask(grey_morphological_denoising(filtered))[0]
df = DistanceFields(shape=volume.shape, seed_point=(50, 50, 50))
pressure = df.pressure_field(mask)
thrust = df.thrust_field(mask)

# Generate skeleton
maxima = df.find_thrust_maxima(thrust, mask, order=2)
skeleton = generate_skeleton_from_seed(maxima, (50, 50, 50), pressure, mask, volume.shape)

# Create graph and export SWC
graph = Graph(create_maxima_image(skeleton, volume.shape), (50, 50, 50))
graph.calculate_mst()
graph.save_to_swc("output.swc", pressure)
```

### Configuration Parameters

Fine-tune the reconstruction with command-line arguments:

```bash
cend \
  --data_dir ./data \
  --output_dir ./results_swc \
  --filter_type yang \
  --sigma_min 1.0 \
  --sigma_max 2.0 \
  --sigma_step 0.5 \
  --neuron_threshold 0.05 \
  --pruning_threshold 10 \
  --maximas_min_dist 2 \
  --smoothing_factor 0.8 \
  --num_points_per_branch 15 \
  --parallel_jobs 2
```

#### Parameter Reference

| Parameter                 | Description                                 | Default         |
| ------------------------- | ------------------------------------------- | --------------- |
| `--data_dir`              | Input directory containing TIFF stacks      | `./data`        |
| `--output_dir`            | Output directory for SWC files              | `./results_swc` |
| `--filter_type`           | Tubular filter type                         | `yang`          |
| `--sigma_min`             | Minimum sigma for multi-scale filtering     | `1.0`           |
| `--sigma_max`             | Maximum sigma for multi-scale filtering     | `2.0`           |
| `--sigma_step`            | Sigma increment step                        | `0.5`           |
| `--neuron_threshold`      | Tubularity threshold for segmentation       | `0.05`          |
| `--pruning_threshold`     | Branch length threshold (nodes, 0=disabled) | `0`             |
| `--maximas_min_dist`      | Window size for local maxima detection      | `2`             |
| `--smoothing_factor`      | Spline smoothing factor                     | `0.8`           |
| `--num_points_per_branch` | Points per branch in output                 | `15`            |
| `--parallel_jobs`         | Number of parallel processes                | `2`             |

### Evaluation with DiademMetric

Compare reconstructions against gold standards:

```bash
make all
```

This will:

1. Run DiademMetric on all SWC files
2. Generate a scores CSV file with metrics
3. Save results to `./scores/scores.csv`

Evaluate a specific parameter configuration:

```bash
make eval FILTER_TYPE=yang SIG_MIN=1.0 SIG_MAX=2.0 SIG_STEP=0.5 \
  NEURON_THRESHOLD=0.05 PRUNING_THRESHOLD=10 SMOOTHING_FACTOR=0.8
```

## Project Structure

```
cend/
├── src/cend/                    # Main package source
│   ├── core/                    # Core algorithms
│   │   ├── filters.py           # Tubular filtering (Yang, Frangi, Kumar)
│   │   ├── segmentation.py      # Thresholding and denoising
│   │   ├── distance_fields.py   # Pressure/thrust fields
│   │   ├── skeletonization.py   # Dijkstra-based tracing
│   │   └── vector_fields.py     # VFC utilities
│   ├── structures/              # Data structures
│   │   ├── graph.py             # MST and graph operations
│   │   └── swc.py               # SWC file format
│   ├── io/                      # Input/output
│   │   └── image.py             # TIFF stack loading
│   ├── processing/              # High-level pipelines
│   │   ├── multiscale.py        # Multi-scale filtering
│   │   └── pipeline.py          # Main reconstruction pipeline
│   ├── visualization/           # Visualization tools
│   │   └── rendering.py         # 3D rendering with Open3D
│   └── cli/                     # Command-line interface
│       └── commands.py          # CLI entry points
├── notebooks/                   # Jupyter notebooks
│   ├── distance_fields.ipynb    # Algorithm demonstrations
│   ├── tests_analysis.ipynb     # Results analysis
│   └── research/                # Experimental code
│       └── cevd.py              # Alternative MVEF algorithm
├── tests/                       # Unit tests
│   └── fixtures/                # Test data
├── data/                        # Input TIFF stacks
│   ├── OP_1/ ... OP_9/         # Image datasets
│   └── GoldStandardReconstructions/
├── metrics/DiademMetric/        # Evaluation tool
├── scripts/                     # Utility scripts
├── pyproject.toml               # Package configuration
├── makefile                     # Evaluation automation
└── README.md                    # This file
```

└── metrics/DiademMetric/ # Evaluation tool

```

## Algorithm Pipeline

The reconstruction follows these steps:

1. **Image Loading**: Load 3D TIFF stack from directory
2. **Preprocessing**: Gaussian filtering and minimum filtering for noise reduction
3. **Tubular Filtering**: Multi-scale filtering to enhance neuron structures
4. **Segmentation**: Adaptive thresholding with morphological cleanup
5. **Distance Fields**: Compute pressure and thrust fields for directional guidance
6. **Skeletonization**: Generate skeleton from seed point using field-guided tracing
7. **Graph Construction**: Build graph from skeleton with 26-connectivity
8. **MST Computation**: Calculate minimum spanning tree from root node
9. **Pruning**: Remove short spurious branches (optional)
10. **Smoothing**: Apply spline interpolation for smooth curves
11. **Export**: Save result in SWC format

## Theoretical Background

### Hessian-Based Tubular Filtering

The foundation of neuron enhancement in CEND relies on **Hessian-based tubular filtering**, which exploits the second-order structure of the image intensity function to detect elongated, tube-like structures.

#### The Hessian Matrix

For a 3D image $I(x, y, z)$, the **Hessian matrix** $\mathbf{H}$ at scale $\sigma$ is defined as:

$$
\mathbf{H}_\sigma = \begin{bmatrix}
I_{xx} & I_{xy} & I_{xz} \\
I_{xy} & I_{yy} & I_{yz} \\
I_{xz} & I_{yz} & I_{zz}
\end{bmatrix} * G_\sigma
$$

where $I_{ij}$ denotes the second partial derivative of the image with respect to coordinates $i$ and $j$, and $G_\sigma$ is a Gaussian kernel with standard deviation $\sigma$ used to compute derivatives at different scales.

#### Eigenvalue Analysis

The eigenvalues $\lambda_1, \lambda_2, \lambda_3$ (ordered such that $|\lambda_1| \leq |\lambda_2| \leq |\lambda_3|$) of the Hessian matrix encode crucial geometric information:

- **Tubular structures (neurites)**: $\lambda_1 \approx 0$, $\lambda_2 \ll 0$, $\lambda_3 \ll 0$
  - The eigenvector corresponding to $\lambda_1$ points along the tube axis
  - The eigenvectors for $\lambda_2$ and $\lambda_3$ span the cross-sectional plane

- **Planar structures**: $\lambda_1 \approx 0$, $\lambda_2 \approx 0$, $\lambda_3 \ll 0$

- **Blob-like structures**: $\lambda_1 \ll 0$, $\lambda_2 \ll 0$, $\lambda_3 \ll 0$

- **Background (uniform regions)**: $\lambda_1 \approx 0$, $\lambda_2 \approx 0$, $\lambda_3 \approx 0$

#### Yang Tubularity Measure

The **Yang tubularity function** $f_{\text{Yang}}(\mathbf{H})$ is designed to produce high responses for tubular structures while suppressing other geometries:

$$
f_{\text{Yang}}(\lambda_1, \lambda_2, \lambda_3) = \begin{cases}
|\lambda_2| \cdot |\lambda_3| & \text{if } |\lambda_1| < \tau \text{ and } \lambda_2 < 0 \text{ and } \lambda_3 < 0 \\
0 & \text{otherwise}
\end{cases}
$$

where $\tau$ is a threshold parameter (`neuron_threshold`) that controls sensitivity.

**Key properties:**
- Only responds to structures with two large negative eigenvalues (indicating tubular geometry)
- Requires the smallest eigenvalue to be near zero (elongation condition)
- Magnitude of response correlates with tube contrast and width
- Scale-normalized by multiplying with $\sigma^2$ to enable multi-scale detection

#### Multi-Scale Analysis

Neurons exhibit varying widths throughout their structure. To capture this variability, CEND applies tubular filtering at multiple scales $\sigma \in [\sigma_{\min}, \sigma_{\max}]$:

$$
R(x, y, z) = \max_{\sigma \in \Sigma} \sigma^2 \cdot f_{\text{Yang}}(\mathbf{H}_\sigma(x, y, z))
$$

The scale normalization factor $\sigma^2$ ensures that responses are comparable across scales, and the maximum operation selects the scale that best matches the local neurite diameter.

### Distance Field-Based Tracing

After tubular filtering and segmentation, CEND employs **distance fields** to guide the skeletonization process. This approach ensures that the extracted centerline follows the medial axis of the neuron structure.

#### Pressure Field

The **pressure field** $P$ is computed using the Euclidean Distance Transform (EDT) on the binary segmentation mask:

$$
P(x, y, z) = \text{EDT}(\mathcal{M}(x, y, z))
$$

where $\mathcal{M}$ is the binary mask of neuron voxels. The pressure field represents the distance from each interior point to the nearest boundary, forming "ridges" along the centerline of tubular structures.

**Properties:**
- Local maxima of $P$ correspond to centerline points
- Magnitude indicates tube thickness
- Gradient $\nabla P$ points away from boundaries toward the medial axis

#### Thrust Field

The **thrust field** $T$ captures directional information by computing the gradient magnitude of the pressure field:

$$
T(x, y, z) = \|\nabla P(x, y, z)\|
$$

The thrust field has high values in regions where the pressure field changes rapidly, which typically occurs near the boundaries and transitions between structures.

**Usage in CEND:**
- Local maxima of the thrust field identify branching points and seed locations
- Combined with the pressure field, it guides the path-finding algorithm
- Smoothing with Gaussian filtering ($\sigma \approx 1-2$) removes noise while preserving structure

#### Skeleton Generation

Starting from a user-specified root coordinate, the algorithm performs a **field-guided path search**:

1. Find local maxima in the thrust field (candidate seed points)
2. From each seed point, trace paths following the gradient of the pressure field
3. Paths naturally converge toward centerlines due to the ridge structure
4. Skeletonize the resulting binary image using morphological thinning

This approach is more robust than pure morphological skeletonization because it:
- Leverages intensity information from the original image
- Reduces sensitivity to boundary irregularities
- Produces smoother, more anatomically plausible centerlines

### Graph-Based Representation

The final skeleton is represented as a **graph** $G = (V, E)$ where:
- Vertices $V$ correspond to skeleton voxels
- Edges $E$ connect 26-connected neighbors (full 3D connectivity)

A **Minimum Spanning Tree (MST)** rooted at the soma location is computed using Prim's algorithm, which:
- Ensures a tree structure (no cycles)
- Minimizes total edge length
- Provides a hierarchical representation suitable for SWC export

Optional **branch pruning** removes terminal branches shorter than a specified threshold, eliminating spurious structures caused by noise or segmentation artifacts.

## SWC Format

Output files follow the standard SWC format used in computational neuroscience:

```

# n T x y z R P

1 2 10.5 20.3 5.0 1.0 -1
2 2 11.2 21.1 5.5 1.0 1
3 2 12.0 22.5 6.0 1.0 2
...

```

Where:
- `n`: Node ID
- `T`: Structure type (2 = axon)
- `x, y, z`: 3D coordinates
- `R`: Radius estimate
- `P`: Parent node ID (-1 for root)

## Example Output

```

Processing image 1: OP_1
✓ Loaded volume: (64, 512, 512)
✓ Applied multi-scale filtering (σ=1.0-2.0)
✓ Generated distance fields
✓ Created skeleton (1247 points)
✓ Built graph (1247 nodes, 1246 edges)
✓ Computed MST
✓ Pruned 23 short branches
✓ Exported to results_swc/OP_1.swc
DiademScore: 0.8734

````

## Tips for Best Results

> [!TIP]
> Start with default parameters and adjust based on your data characteristics:
> - **Dense structures**: Increase `sigma_max` to 3.0-4.0
> - **Noisy images**: Lower `neuron_threshold` to 0.03-0.04
> - **Spurious branches**: Enable pruning with threshold 10-20
> - **Memory issues**: Reduce `parallel_jobs` to 1

> [!NOTE]
> The pipeline requires root coordinates to be specified for each dataset in `pipeline_swc.py`. Update the `roots` list if processing custom data.

> [!WARNING]
> Processing large volumes (>1GB per image) may require significant RAM. Monitor memory usage and reduce `parallel_jobs` if needed.

## Development

### Running Tests

Execute test notebooks for algorithm validation:

```bash
jupyter notebook tests_analysis.ipynb
````

### Visualization

Use the provided utilities to visualize results:

```python
from visualize import plot_projections, render_mesh_from_volume
import numpy as np

# Load your volume
volume = ...

# Plot 2D projections
plot_projections(volume, aggregations='max', axes=0, cmaps='gray')

# 3D rendering
render_mesh_from_volume(volume, threshold=0.1)
```

## References

This implementation is based on distance field tracing methods for neuron reconstruction:

- Yang et al. (2013): "A distance-field based automatic neuron tracing method"
- Frangi et al. (1998): "Multiscale vessel enhancement filtering"

## Acknowledgments

Evaluation uses the DIADEM Metric developed for the DIADEM Challenge:

- http://diademchallenge.org/

---

**Note**: This project is designed for research purposes in computational neuroscience and bioimage analysis.
