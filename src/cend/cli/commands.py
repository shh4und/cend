"""
Command-line interface for CEND neuron reconstruction.
"""

import sys

from ..processing.pipeline import main as pipeline_main


def main():
    """Main CLI entry point."""
    try:
        pipeline_main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
