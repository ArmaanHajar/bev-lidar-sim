"""Sensor layer: scene geometry primitives and the 2D LiDAR."""

from .lidar import Lidar2D, ScanResult
from .scene import Box, Scene, build_demo_scene

__all__ = ["Lidar2D", "ScanResult", "Box", "Scene", "build_demo_scene"]
