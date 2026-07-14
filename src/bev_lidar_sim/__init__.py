"""BEV LiDAR and traffic simulation package.

Layered layout: `sensors` (scene + LiDAR), `maps` (RoadGraph schema),
`sim` (traffic core + scenarios), `render` (pygame live / matplotlib
stills), `cli` (drive / demo entry points).

This top level re-exports the headless public surface; rendering and the
CLIs are imported explicitly so plain simulation never pulls in pygame or
matplotlib.
"""

from .maps.roadgraph import ConnectorDef, LaneDef, RoadGraph, RoadNode
from .sensors.lidar import Lidar2D, ScanResult
from .sensors.scene import Box, Scene
from .sim.arterial import ArterialSimulator
from .sim.city import CitySimulator, build_city_roadgraph

__all__ = ["ArterialSimulator", "CitySimulator", "build_city_roadgraph",
           "Lidar2D", "ScanResult", "RoadGraph", "RoadNode", "LaneDef",
           "ConnectorDef", "Box", "Scene"]
