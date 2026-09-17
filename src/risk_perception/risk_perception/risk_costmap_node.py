#!/usr/bin/env python3
"""
risk_costmap_node.py  --  perception -> Nav2 costmap bridge

Rasterizes the persistent world model (object_tracker_node's confirmed
tracks, /risk_perception/world_objects) into a nav_msgs/OccupancyGrid risk
field that nav2_risk_layer::RiskLayer overlays onto the Nav2 master costmap.

Each object contributes a filled disc of its own risk score (from
risk_visualization.risk_score_from_label) out to its own footprint radius,
then an exponential falloff (`alpha` per grid cell outward) beyond that, so
nearby cells still see a graded warning rather than a hard edge. Where two
objects' fields overlap, a cell takes the MAX -- one high-risk person
standing next to a chair should read as "risky", not "off-scale".

The falloff distance is capped by `max_falloff_radius_m`, independent of how
the exponential decay is tuned. This is a hard-learned lesson: `alpha: 0.90`
(the old default) looks like a gentle, tasteful decay, but on a 10cm grid it
takes 22-37 cells -- 2.2 to 3.7 METERS -- for any object's risk to fall
under a typical `min_risk`, so a room with a dozen objects ends up wall-to-
wall risk. `alpha: 0.25` is what actually produces the "graded warning a few
tens of cm out" this docstring describes; the cap exists so a future retune
of `alpha`/`min_risk` cannot reintroduce a room-filling halo no matter what
value gets picked -- the paint radius is bounded independently of that math.

Publishes on a fixed timer (`publish_rate`) rather than per-detection, and
ALWAYS republishes the full grid, including cells that dropped back to zero
risk. That matters: nav2_risk_layer's updateBounds claims this grid's entire
footprint every cycle, so Nav2 fully resets and recombines that region each
time -- an accurate zero here is what lets risk actually clear from the
Nav2 costmap as an object moves away, not just accumulate.

Grid values are risk*100 as int8 (0-100), matching what nav2_risk_layer
expects (values below its `min_risk_value`, or negative, are treated as
free space).

class_id contract: object_tracker_node publishes its belief state inline as
"label|pmov=0.90|pmot=0.05", so everything downstream must split on "|" before
looking the label up -- see _paint_object.
"""

import math
import time
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection3DArray

from risk_perception.risk_visualization import risk_score_from_label
from risk_perception.debug_log import open_latency_csv, log_latency


class RiskCostmapNode(Node):
    def __init__(self) -> None:
        super().__init__("risk_costmap_node")

        self.declare_parameter("input_topic", "/risk_perception/world_objects")
        self.declare_parameter("output_topic", "/risk_costmap")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("resolution", 0.10)
        self.declare_parameter("width_m", 10.0)
        self.declare_parameter("height_m", 10.0)
        self.declare_parameter("origin_x", -5.0)
        self.declare_parameter("origin_y", -5.0)
        self.declare_parameter("alpha", 0.90)
        self.declare_parameter("min_risk", 0.02)
        self.declare_parameter("publish_rate", 5.0)
        self.declare_parameter("footprint_pad", 0.05)
        self.declare_parameter("min_footprint_radius_m", 0.20)
        self.declare_parameter("max_falloff_radius_m", 0.30)
        # Per-node compute-time breakdown (off unless set) -- see
        # gdino_detector_node.py's own comment on this same param.
        self.declare_parameter("debug_log_dir", "")

        gp = self.get_parameter
        self.lat_writer, self.lat_file = open_latency_csv(
            self, str(gp("debug_log_dir").value))
        input_topic = str(gp("input_topic").value)
        output_topic = str(gp("output_topic").value)
        self.map_frame = str(gp("map_frame").value)
        self.resolution = float(gp("resolution").value)
        self.width_m = float(gp("width_m").value)
        self.height_m = float(gp("height_m").value)
        self.origin_x = float(gp("origin_x").value)
        self.origin_y = float(gp("origin_y").value)
        self.alpha = min(0.999, max(0.01, float(gp("alpha").value)))
        self.min_risk = float(gp("min_risk").value)
        self.footprint_pad = float(gp("footprint_pad").value)
        self.min_footprint_radius = float(gp("min_footprint_radius_m").value)
        # Hard cap on the graded falloff beyond an object's footprint -- see
        # the module docstring. Deliberately NOT derived from alpha/min_risk
        # (that derivation is exactly what let the halo balloon to 2-4m): this
        # bounds the paint radius directly, in metres, regardless of how the
        # decay curve is tuned.
        self.max_falloff_radius = float(gp("max_falloff_radius_m").value)

        self.width_cells = max(1, int(round(self.width_m / self.resolution)))
        self.height_cells = max(1, int(round(self.height_m / self.resolution)))

        self.latest_objects: Optional[Detection3DArray] = None

        self.create_subscription(Detection3DArray, input_topic, self._objects_cb, 10)
        # TRANSIENT_LOCAL to match nav2_risk_layer's transient_local+reliable
        # subscription (risk_layer.cpp) -- a VOLATILE publisher is QoS-
        # incompatible with it and the layer would silently receive nothing.
        # Also hands a late-joining RViz the last grid immediately.
        grid_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(OccupancyGrid, output_topic, grid_qos)

        publish_rate = max(0.1, float(gp("publish_rate").value))
        self.create_timer(1.0 / publish_rate, self._publish_grid)

        self.get_logger().info(
            f"Rasterizing {input_topic} -> {output_topic} "
            f"({self.width_cells}x{self.height_cells} cells @ {self.resolution} m, "
            f"origin ({self.origin_x}, {self.origin_y}))")

    def _objects_cb(self, msg: Detection3DArray) -> None:
        self.latest_objects = msg

    def _publish_grid(self) -> None:
        t0 = time.perf_counter()
        grid = np.zeros((self.height_cells, self.width_cells), dtype=np.float32)

        if self.latest_objects is not None:
            for detection in self.latest_objects.detections:
                self._paint_object(grid, detection)

        data = np.clip(np.round(grid * 100.0), 0, 100).astype(np.int8)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        info = MapMetaData()
        info.resolution = self.resolution
        info.width = self.width_cells
        info.height = self.height_cells
        info.origin.position.x = self.origin_x
        info.origin.position.y = self.origin_y
        info.origin.orientation.w = 1.0
        msg.info = info

        msg.data = data.flatten().tolist()

        self.publisher.publish(msg)
        log_latency(self.lat_writer, self.lat_file, time.perf_counter() - t0)

    def _paint_object(self, grid: np.ndarray, detection) -> None:
        if not detection.results:
            return

        hypothesis = detection.results[0].hypothesis
        # the tracker encodes its belief state into class_id as
        # "label|pmov=..|pmot=.."; risk is keyed on the bare label, and the
        # unsplit string would silently miss CLASS_BASE_RISK and fall back to
        # 0.40 for everything.
        label = str(hypothesis.class_id).split("|")[0]
        score = float(hypothesis.score)
        risk = risk_score_from_label(label, score)

        if risk < self.min_risk:
            return

        cx = float(detection.bbox.center.position.x)
        cy = float(detection.bbox.center.position.y)

        footprint_radius = max(
            self.min_footprint_radius,
            max(float(detection.bbox.size.x), float(detection.bbox.size.y)) / 2.0,
        ) + self.footprint_pad

        paint_radius_m = footprint_radius + self.max_falloff_radius
        gx = (cx - self.origin_x) / self.resolution
        gy = (cy - self.origin_y) / self.resolution
        r_cells = paint_radius_m / self.resolution

        i_min = max(0, int(math.floor(gx - r_cells)))
        i_max = min(self.width_cells, int(math.ceil(gx + r_cells)) + 1)
        j_min = max(0, int(math.floor(gy - r_cells)))
        j_max = min(self.height_cells, int(math.ceil(gy + r_cells)) + 1)

        if i_min >= i_max or j_min >= j_max:
            return

        ii, jj = np.meshgrid(np.arange(i_min, i_max), np.arange(j_min, j_max))
        cell_x = self.origin_x + (ii + 0.5) * self.resolution
        cell_y = self.origin_y + (jj + 0.5) * self.resolution
        dist = np.hypot(cell_x - cx, cell_y - cy)

        beyond = np.clip((dist - footprint_radius) / self.resolution, 0.0, None)
        value = risk * (self.alpha ** beyond)
        value[dist <= footprint_radius] = risk
        value[value < self.min_risk] = 0.0

        region = grid[j_min:j_max, i_min:i_max]
        np.maximum(region, value, out=region)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RiskCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
