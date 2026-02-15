# Migration to src-layout Structure

## Summary

The CEND project has been successfully restructured from a flat-layout to a professional src-based package structure. This migration improves modularity, testability, and follows Python packaging best practices.

## What Changed

### Architecture Improvements

1. **Modular Structure**: Split large monolithic files into focused modules

   - `dst_fields.py` (625 LOC) → 4 modules: `filters.py`, `segmentation.py`, `distance_fields.py`, `skeletonization.py`
   - Extracted shared filter implementations to eliminate duplication
   - Separated concerns: core algorithms, processing, I/O, visualization

2. **Professional Package Layout**: Adopted src-layout structure

   - All source code now in `src/cend/`
   - Proper `__init__.py` files with public API exports
   - CLI entry point via `pyproject.toml` scripts

3. **Clean Dependencies**: Updated `pyproject.toml`
   - Removed version pinning (uses ranges: `>=`)
   - Added optional dependencies (`dev`, `notebooks`)
   - Configured for setuptools with `packages.find`

### File Locations

#### New Structure

```
src/cend/
├── __init__.py              # Package entry point
├── core/                    # Core algorithms
│   ├── filters.py           # Tubular filtering (Yang, Frangi, Kumar, Sato)
│   ├── segmentation.py      # Thresholding and morphological ops
│   ├── distance_fields.py   # Pressure/thrust fields
│   ├── skeletonization.py   # Dijkstra-based tracing
│   └── vector_fields.py     # VFC utilities (from vfc.py)
├── structures/              # Data structures
│   ├── graph.py             # MST computation (from graphs.py)
│   └── swc.py               # SWC format handling
├── io/                      # Input/output
│   └── image.py             # TIFF I/O (from image_io.py)
├── processing/              # High-level pipelines
│   ├── multiscale.py        # Multi-scale filtering orchestration
│   └── pipeline.py          # Main pipeline (from pipeline_swc.py)
├── visualization/           # Visualization
│   └── rendering.py         # 3D rendering (from visualize.py)
└── cli/                     # Command-line interface
    └── commands.py          # CLI entry point
```

#### Moved Files

- **Experimental code**: `cevd.py` → `notebooks/research/cevd.py`
- **Notebooks**: `*.ipynb` → `notebooks/`
- **Old modules**: Backed up in `_old_modules/`

## How to Use

### Installation

```bash
# Install in development mode
pip install -e .

# With optional dependencies
pip install -e ".[dev,notebooks]"
```

### Command Line

The CLI command is now available as `cend`:

```bash
# Process all images
cend --data_dir ./data --output_dir ./results_swc

# Process single image
cend --image_index 1

# With custom parameters
cend --filter_type yang --sigma_min 1.0 --sigma_max 2.0
```

### Python API

Import from the new package structure:

```python
# Core functionality
from cend import DistanceFields, Graph, SWCFile, load_3d_volume, multiscale_filtering

# Specific modules
from cend.core import filters, segmentation, skeletonization
from cend.core.filters import yang_tubularity, frangi_vesselness
from cend.core.segmentation import adaptive_mean_mask, morphological_denoising
from cend.structures import Graph, SWCFile
from cend.io import load_3d_volume, save_3d_volume

# Example usage
volume = load_3d_volume("./data/OP_1")
filtered = multiscale_filtering(volume, sigma_range=(1, 2, 0.5))
mask, threshold = adaptive_mean_mask(filtered)
```

### Notebooks

Update notebook imports to use the new structure:

```python
# Old imports (no longer work)
# from dst_fields import DistanceFields
# from image_io import load_3d_volume

# New imports
from cend.core import DistanceFields
from cend.io import load_3d_volume
from cend.visualization import plot_projections
```

## Benefits

1. **Better Organization**: Clear separation of concerns

   - Core algorithms independent of I/O
   - Easy to locate and modify specific functionality

2. **Improved Testability**: Smaller, focused modules are easier to test

   - Each module has a single responsibility
   - Reduced coupling between components

3. **Professional Package**: Follows Python best practices

   - Standard src-layout structure
   - Proper entry points and CLI
   - Clean dependency management

4. **Eliminate Duplication**: Filter implementations extracted to shared module

   - ~200 LOC saved by removing duplicate code
   - Single source of truth for algorithms

5. **Better Documentation**: Structured exports in `__init__.py`
   - Clear public API surface
   - Better IDE autocomplete support

## Migration Checklist

- [x] Create src-layout directory structure
- [x] Extract shared filter implementations
- [x] Split DistanceFields into 4 modules
- [x] Move existing modules to new locations
- [x] Update all import statements
- [x] Move experimental code to notebooks/research
- [x] Create **init**.py files with public exports
- [x] Create CLI entry point
- [x] Update pyproject.toml for src-layout
- [x] Update .gitignore
- [x] Update README.md with new examples
- [x] Install and test package
- [x] Verify imports work correctly
- [x] Verify CLI command works

## Verification

All tests passed:

```bash
# Imports work
✓ from cend.core import DistanceFields
✓ from cend.structures import Graph
✓ from cend import load_3d_volume

# CLI available
✓ cend --help

# Package installed
✓ pip show cend
```

## Next Steps

1. **Write Unit Tests**: Add tests in `tests/` directory

   ```
   tests/
   ├── test_filters.py
   ├── test_segmentation.py
   ├── test_distance_fields.py
   └── test_graph.py
   ```

2. **Update Notebooks**: Modify notebooks to use new imports

3. **Documentation**: Add docstrings and API documentation

4. **Type Hints**: Complete type annotations for all public functions

5. **Remove Backup**: Delete `_old_modules/` after confirming everything works

## Rollback (if needed)

If you need to revert to the old structure:

```bash
# Restore old files
cp _old_modules/* .

# Revert pyproject.toml
git checkout pyproject.toml

# Uninstall new package
pip uninstall cend
```

## Questions?

- Check the updated README.md for usage examples
- See `src/cend/__init__.py` for the public API
- Old code is preserved in `_old_modules/` for reference
