"""Backward-compatible wrapper for the driving simulator CLI."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bev_lidar_sim.cli.drive import main


if __name__ == "__main__":
    main()

