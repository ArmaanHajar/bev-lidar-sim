"""Rendering layer: pygame live view (`live`) and matplotlib stills (`stills`).

The two stay separate on purpose — they serve different artifacts.
Re-exports cover what the driving CLI uses, so `from bev_lidar_sim import
render` keeps working as before the split.
"""

from .live import GAP, HEIGHT, LIDAR_H, W, WORLD_H, draw_lidar, draw_world

__all__ = ["draw_world", "draw_lidar", "W", "HEIGHT", "WORLD_H", "LIDAR_H",
           "GAP"]
