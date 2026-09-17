#!/usr/bin/env python3

from collections import OrderedDict
from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from risk_perception.model_adapters.sam2_adapter import (
    Sam2Adapter,
)

from risk_perception.risk_visualization import (
    risk_score_from_label,
    risk_to_bgr,
)
from risk_perception.debug_log import open_latency_csv, log_latency

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.35
font_thickness = 1
box_thickness = 2
text_color = (255, 255, 255)
padding = 3

# annotation = f"{labels[index]} {risk_score:.2f}"

class Sam2SegmenterNode(Node):
    """
    Segment Grounding DINO detections using SAM 2 box prompts.

    Images are cached by their exact ROS timestamp. The incoming
    Detection2DArray retains the original image timestamp, so the node
    retrieves the frame that actually generated the detections.
    """

    def __init__(self) -> None:
        super().__init__("sam2_segmenter")

        self.declare_parameter(
            "image_topic",
            "/camera/camera/color/image_raw",
        )
        self.declare_parameter(
            "detections_topic",
            "/risk_perception/detections_2d",
        )
        self.declare_parameter(
            "mask_topic",
            "/risk_perception/instance_mask",
        )
        self.declare_parameter(
            "segmentation_image_topic",
            "/risk_perception/segmentation_image",
        )

        self.declare_parameter(
            "model_config",
            "configs/sam2.1/sam2.1_hiera_s.yaml",
        )
        self.declare_parameter(
            "checkpoint_path",
            "",
        )
        self.declare_parameter(
            "device",
            "cuda",
        )
        self.declare_parameter(
            "image_cache_size",
            60,
        )
        self.declare_parameter(
            "overlay_alpha",
            0.35,
        )
        # Per-node compute-time breakdown (off unless set) -- see
        # gdino_detector_node.py's own comment on this same param.
        self.declare_parameter("debug_log_dir", "")
        self.lat_writer, self.lat_file = open_latency_csv(
            self, str(self.get_parameter("debug_log_dir").value))

        image_topic = str(
            self.get_parameter("image_topic").value
        )
        detections_topic = str(
            self.get_parameter("detections_topic").value
        )
        mask_topic = str(
            self.get_parameter("mask_topic").value
        )
        segmentation_image_topic = str(
            self.get_parameter(
                "segmentation_image_topic"
            ).value
        )

        model_config = str(
            self.get_parameter("model_config").value
        )
        checkpoint_path = str(
            self.get_parameter("checkpoint_path").value
        )
        device = str(
            self.get_parameter("device").value
        )

        self.image_cache_size = max(
            5,
            int(
                self.get_parameter(
                    "image_cache_size"
                ).value
            ),
        )

        self.overlay_alpha = float(
            self.get_parameter(
                "overlay_alpha"
            ).value
        )
        self.overlay_alpha = min(
            1.0,
            max(0.0, self.overlay_alpha),
        )

        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"SAM 2 checkpoint not found: {checkpoint_path}"
            )

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested, but CUDA is unavailable"
            )

        self.get_logger().info(
            f"Loading SAM 2 on {device}"
        )

        load_start = time.perf_counter()

        self.segmenter = Sam2Adapter(
            model_config=model_config,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        self.get_logger().info(
            "SAM 2 loaded in "
            f"{time.perf_counter() - load_start:.2f} s"
        )

        self.bridge = CvBridge()

        # Key: (sec, nanosec)
        # Value: sensor_msgs/Image
        self.image_cache = OrderedDict()

        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.detection_subscription = self.create_subscription(
            Detection2DArray,
            detections_topic,
            self.detection_callback,
            10,
        )

        self.mask_publisher = self.create_publisher(
            Image,
            mask_topic,
            10,
        )

        self.segmentation_image_publisher = self.create_publisher(
            Image,
            segmentation_image_topic,
            10,
        )

        self.processed_count = 0
        self.cache_miss_count = 0

        self.get_logger().info(
            f"Image topic: {image_topic}"
        )
        self.get_logger().info(
            f"Detection topic: {detections_topic}"
        )
        self.get_logger().info(
            f"Mask output: {mask_topic}"
        )
        self.get_logger().info(
            f"Segmentation output: {segmentation_image_topic}"
        )
        self.get_logger().info(
            f"Image cache size: {self.image_cache_size}"
        )

    @staticmethod
    def stamp_key(msg) -> tuple[int, int]:
        return (
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
        )

    def image_callback(self, msg: Image) -> None:
        key = self.stamp_key(msg)

        self.image_cache[key] = msg
        self.image_cache.move_to_end(key)

        while len(self.image_cache) > self.image_cache_size:
            self.image_cache.popitem(last=False)

    def detection_callback(
        self,
        msg: Detection2DArray,
    ) -> None:
        key = self.stamp_key(msg)

        image_msg = self.image_cache.get(key)

        if image_msg is None:
            self.cache_miss_count += 1

            self.get_logger().warning(
                "Could not find the RGB frame matching detection "
                f"timestamp {key}. "
                f"Cache size={len(self.image_cache)}, "
                f"misses={self.cache_miss_count}"
            )
            return

        try:
            image_bgr = self.bridge.imgmsg_to_cv2(
                image_msg,
                desired_encoding="bgr8",
            )
        except CvBridgeError as error:
            self.get_logger().error(
                f"RGB conversion failed: {error}"
            )
            return

        height, width = image_bgr.shape[:2]

        boxes = []
        labels = []
        detection_scores = []

        for detection in msg.detections:
            center_x = float(
                detection.bbox.center.position.x
            )
            center_y = float(
                detection.bbox.center.position.y
            )
            size_x = float(detection.bbox.size_x)
            size_y = float(detection.bbox.size_y)

            x1 = center_x - size_x / 2.0
            y1 = center_y - size_y / 2.0
            x2 = center_x + size_x / 2.0
            y2 = center_y + size_y / 2.0

            x1 = max(0.0, min(x1, width - 1.0))
            x2 = max(0.0, min(x2, width - 1.0))
            y1 = max(0.0, min(y1, height - 1.0))
            y2 = max(0.0, min(y2, height - 1.0))

            if x2 <= x1 or y2 <= y1:
                continue

            label = "object"
            detection_score = 0.0

            if detection.results:
                hypothesis = detection.results[0].hypothesis
                label = str(hypothesis.class_id)
                detection_score = float(hypothesis.score)

            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            detection_scores.append(detection_score)

        if len(boxes) == 0:
            self.publish_empty_result(
                image_msg=image_msg,
                image_bgr=image_bgr,
            )
            return

        boxes_array = np.asarray(
            boxes,
            dtype=np.float32,
        )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        try:
            inference_start = time.perf_counter()

            result = self.segmenter.segment(
                image_rgb=image_rgb,
                boxes_xyxy=boxes_array,
            )

            inference_seconds = (
                time.perf_counter() - inference_start
            )
            log_latency(self.lat_writer, self.lat_file, inference_seconds)

        except Exception as error:
            self.get_logger().error(
                f"SAM 2 inference failed: {error}"
            )
            return

        if len(result.masks) != len(boxes):
            self.get_logger().error(
                "SAM 2 result count does not match box count: "
                f"{len(result.masks)} masks vs {len(boxes)} boxes"
            )
            return

        instance_mask = np.zeros(
            (height, width),
            dtype=np.uint16,
        )

        overlay = image_bgr.copy()

        palette = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 165, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        ]

        for index, mask in enumerate(result.masks):
            instance_id = index + 1

            risk_score = risk_score_from_label(labels[index], detection_scores[index], )
            color = risk_to_bgr(risk_score)

            mask = np.asarray(mask, dtype=bool)

            if mask.shape != (height, width):
                self.get_logger().warning(
                    f"Skipping mask {index}: "
                    f"shape={mask.shape}, expected={(height, width)}"
                )
                continue

            instance_mask[mask] = instance_id

            overlay_pixels = overlay[mask].astype(
                np.float32
            )
            color_array = np.asarray(
                color,
                dtype=np.float32,
            )

            overlay[mask] = (
                (1.0 - self.overlay_alpha) * overlay_pixels
                + self.overlay_alpha * color_array
            ).astype(np.uint8)

            x1, y1, x2, y2 = [
                int(round(value))
                for value in boxes[index]
            ]

            # bbox
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                box_thickness,
            )

            # Clean presentation label.
            # Debug version:
            # annotation = f"{labels[index]} risk {risk_score:.2f} DINO {detection_scores[index]:.2f}"

            # Presentation version:
            annotation = f"{labels[index]} {risk_score:.2f}"

            # text size
            (text_w, text_h), baseline = cv2.getTextSize(
                annotation,
                font,
                font_scale,
                font_thickness,
            )

            # Keep label inside image boundary.
            tx = min(
                max(0, x1),
                max(0, width - text_w - 2 * padding - 1),
            )

            ty = y1 - 6

            # If label would go above the image, put it inside/below the bbox.
            if ty - text_h - padding < 0:
                ty = min(height - baseline - padding - 1, y1 + text_h + 2 * padding)

            # background rectangle for text
            cv2.rectangle(
                overlay,
                (tx, ty - text_h - padding),
                (tx + text_w + 2 * padding, ty + baseline + padding),
                color,
                thickness=-1,
            )

            # white text
            cv2.putText(
                overlay,
                annotation,
                (tx + padding, ty),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
            # cv2.putText(
            #     overlay,
            #     annotation,
            #     (x1, max(22, y1 - 8)),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     color,
            #     font_thickness,
            #     cv2.LINE_AA,
            # )

        self.publish_results(
            image_msg=image_msg,
            instance_mask=instance_mask,
            overlay=overlay,
        )

        self.processed_count += 1

        if self.processed_count % 10 == 0:
            self.get_logger().info(
                f"Segmented {len(boxes)} objects in "
                f"{inference_seconds * 1000.0:.1f} ms"
            )

    def publish_empty_result(
        self,
        image_msg: Image,
        image_bgr: np.ndarray,
    ) -> None:
        height, width = image_bgr.shape[:2]

        instance_mask = np.zeros(
            (height, width),
            dtype=np.uint16,
        )

        self.publish_results(
            image_msg=image_msg,
            instance_mask=instance_mask,
            overlay=image_bgr,
        )

    def publish_results(
        self,
        image_msg: Image,
        instance_mask: np.ndarray,
        overlay: np.ndarray,
    ) -> None:
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(
                instance_mask,
                encoding="mono16",
            )
            mask_msg.header = image_msg.header
            self.mask_publisher.publish(mask_msg)

            overlay_msg = self.bridge.cv2_to_imgmsg(
                overlay,
                encoding="bgr8",
            )
            overlay_msg.header = image_msg.header
            self.segmentation_image_publisher.publish(
                overlay_msg
            )

        except CvBridgeError as error:
            self.get_logger().error(
                f"Output image conversion failed: {error}"
            )


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)

    node = Sam2SegmenterNode()

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
