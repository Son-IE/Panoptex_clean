'''
grounding_dino_adapter.py

'''
#!/usr/bin/env python3

import time
from pathlib import Path
from typing import Optional

import cv2
import rclpy
import torch
import groundingdino
print("GDINO:", groundingdino.__file__)
try:
    from groundingdino import _C
    print("_C: OK")
except ImportError as e:
    print("_C FAILED:", e)
    
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from risk_perception.model_adapters.grounding_dino_adapter import (
    GroundingDinoAdapter,
)
# Pure geometry, deliberately kept out of this file: relation_matching.py has
# no torch/groundingdino import, so its matching logic is unit-testable
# without a GPU or a model checkpoint. See test_relation_matching.py.
from risk_perception.relation_matching import (
    match_relations_to_objects,
    normalize_phrase,
)
# GroundingDINO itself has no NMS (checked against the installed package --
# see nms.py's docstring), so one physical object routinely comes back as
# two or three overlapping boxes in a single frame. See test_nms.py.
from risk_perception.nms import class_aware_nms
from risk_perception.debug_log import open_latency_csv, log_latency


class GroundingDinoDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("gdino_detector")

        self.declare_parameter(
            "image_topic",
            "/camera/camera/color/image_raw",
        )
        self.declare_parameter("config_path", "")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("prompt", "person . chair . monitor . table . mobile robot .")
        self.declare_parameter("box_threshold", 0.35)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("inference_stride", 3)
        self.declare_parameter("device", "cuda")
        # Relation prior (proposed). Empty string = off, identical behavior
        # to before this existed. When set, appended to `prompt` and run as
        # ONE combined GDINO call (not a second inference pass) -- see the
        # relation-prior plan for why a single combined caption was chosen
        # over two separate calls, and what would need re-testing to prove
        # that choice doesn't degrade either phrase type's grounding quality.
        self.declare_parameter("relation_prompt", "")
        # Deliberately lower than a typical NMS threshold: a relation phrase's
        # box is usually a loose region covering both participants, not a
        # tight box around one of them.
        self.declare_parameter("relation_iou_threshold", 0.15)
        # NMS for GDINO's OWN detections -- GroundingDINO has none built in
        # (checked directly, see nms.py). Per-label (class_aware_nms), not
        # class-agnostic: a person and the chair they're sitting on can
        # legitimately share most of a box and must not suppress each
        # other, only two boxes competing for the SAME physical object
        # should. 0.5 is the conventional default; lower if the same object
        # is still coming back doubled at your camera's typical scale.
        self.declare_parameter("nms_iou_threshold", 0.5)
        # GroundingDINO's phrase grounding routinely returns a MERGED span
        # covering two adjacent prompt nouns ("forklift cart", "robot
        # forklift", "table pallet"), because both light up above
        # text_threshold for one box. Those compounds are not in
        # LABEL_CATEGORIES / CLASS_BASE_RISK, so they fall to "unknown" /
        # default risk and -- worse -- churn object_tracker's category-keyed
        # association gate into a new track every frame the label flips. When
        # true, each detection's phrase is reduced to the single prompt term
        # that appears leftmost in it, BEFORE NMS (so the now-identical
        # labels can actually dedupe). Relation phrases are left untouched.
        self.declare_parameter("collapse_compound_labels", True)
        # Per-node compute-time breakdown (off unless set): one row per
        # inference call in <debug_log_dir>/<node_name>_latency_*.csv. See
        # debug_log.open_latency_csv / tools/node_latency_report.py.
        self.declare_parameter("debug_log_dir", "")
        self.lat_writer, self.lat_file = open_latency_csv(
            self, str(self.get_parameter("debug_log_dir").value))

        image_topic = str(
            self.get_parameter("image_topic").value
        )
        config_path = str(
            self.get_parameter("config_path").value
        )
        checkpoint_path = str(
            self.get_parameter("checkpoint_path").value
        )

        self.prompt = str(
            self.get_parameter("prompt").value
        )
        self.box_threshold = float(
            self.get_parameter("box_threshold").value
        )
        self.text_threshold = float(
            self.get_parameter("text_threshold").value
        )
        self.inference_stride = max(
            1,
            int(self.get_parameter("inference_stride").value),
        )
        self.device = str(
            self.get_parameter("device").value
        )
        self.relation_prompt = str(
            self.get_parameter("relation_prompt").value
        )
        self.nms_iou_threshold = float(
            self.get_parameter("nms_iou_threshold").value
        )
        self.relation_iou_threshold = float(
            self.get_parameter("relation_iou_threshold").value
        )
        # Precomputed once, not per-frame: the set of phrases that count as
        # a relation match. Empty when relation_prompt is empty, so nothing
        # is ever classified as a relation and behavior is identical to
        # before this feature existed.
        self.relation_phrases = [
            normalize_phrase(p) for p in self.relation_prompt.split(".")
            if normalize_phrase(p)
        ]
        self.caption = (
            self.prompt if not self.relation_phrases
            else f"{self.prompt} {self.relation_prompt}"
        )

        self.collapse_compound_labels = bool(
            self.get_parameter("collapse_compound_labels").value
        )
        # Object vocabulary in prompt order, longest first so a multi-word
        # entry ("mobile robot") is tried before its tail ("robot").
        self._vocab = sorted(
            {normalize_phrase(t) for t in self.prompt.split(".")
             if normalize_phrase(t)},
            key=len, reverse=True,
        )
        self._vocab_set = set(self._vocab)

        if not Path(config_path).is_file():
            raise FileNotFoundError(
                f"Grounding DINO config not found: {config_path}"
            )

        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Grounding DINO checkpoint not found: "
                f"{checkpoint_path}"
            )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested, but CUDA is unavailable"
            )

        self.get_logger().info(
            f"Loading Grounding DINO on {self.device}"
        )

        load_start = time.perf_counter()

        self.detector = GroundingDinoAdapter(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

        self.get_logger().info(
            "Grounding DINO loaded in "
            f"{time.perf_counter() - load_start:.2f} s"
        )

        self.bridge = CvBridge()
        self.frame_count = 0
        self.inference_count = 0

        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            "/risk_perception/detections_2d",
            10,
        )

        self.image_publisher = self.create_publisher(
            Image,
            "/risk_perception/detection_image",
            10,
        )

        self.get_logger().info(
            f"Subscribing to {image_topic}"
        )
        self.get_logger().info(
            f"Prompt: {self.prompt}"
        )
        if self.relation_phrases:
            self.get_logger().info(
                f"Relation prompt: {self.relation_prompt} "
                f"(iou threshold {self.relation_iou_threshold})"
            )
        self.get_logger().info(
            f"Inference stride: {self.inference_stride}"
        )

    def collapse_label(self, raw: str) -> str:
        """Reduce a GDINO phrase to one prompt term (the leftmost that
        occurs in it); leave single-term and relation phrases untouched.
        'forklift cart' -> 'forklift', 'robot forklift' -> 'robot',
        'table pallet' -> 'table'. No effect if it is already one vocab
        word or matches no vocab word at all."""
        norm = normalize_phrase(raw)
        if not norm or norm in self._vocab_set:
            return norm
        if any(rp in norm or norm in rp for rp in self.relation_phrases):
            return norm
        hits = [(norm.find(term), term) for term in self._vocab
                if term in norm]
        return min(hits)[1] if hits else norm

    def image_callback(self, msg: Image) -> None:
        self.frame_count += 1

        if self.frame_count % self.inference_stride != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except CvBridgeError as error:
            self.get_logger().error(
                f"Image conversion failed: {error}"
            )
            return

        try:
            start_time = time.perf_counter()

            result = self.detector.predict(
                image_bgr=frame,
                prompt=self.caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )

            inference_time = time.perf_counter() - start_time
            self.inference_count += 1
            log_latency(self.lat_writer, self.lat_file, inference_time)

            # Collapse merged prompt spans to a single term BEFORE NMS, so
            # "forklift" and "forklift cart" for one object become the same
            # label and NMS can actually dedupe them.
            if self.collapse_compound_labels:
                result.labels = [
                    self.collapse_label(lbl) for lbl in result.labels
                ]

        except Exception as error:
            self.get_logger().error(
                f"Grounding DINO inference failed: {error}"
            )
            return

        # NMS before anything else touches these boxes: a duplicate here
        # would otherwise get its own SAM2 mask, its own 3D projection, and
        # -- since object_tracker's per-frame assignment is one-to-one --
        # its own brand-new track sitting on top of the real one. See
        # nms.py's docstring.
        if len(result.boxes_xyxy) > 1:
            keep = class_aware_nms(
                result.boxes_xyxy, result.scores, result.labels,
                self.nms_iou_threshold,
            )
            if len(keep) < len(result.boxes_xyxy):
                result.boxes_xyxy = result.boxes_xyxy[keep]
                result.scores = result.scores[keep]
                result.labels = [result.labels[i] for i in keep]

        detection_array = Detection2DArray()
        detection_array.header = msg.header

        annotated = frame.copy()

        # Two passes: (1) classify + build the object detections exactly as
        # before, stashing relation-classified boxes separately instead of
        # publishing them; (2) IoU-match each relation box against this same
        # frame's object boxes and tag the winning object's class_id. Same
        # image, same instant, same coordinate system -- no timestamp sync,
        # no second topic, no 3D projection needed for the relation boxes
        # themselves. See the relation-prior plan for why this replaced the
        # originally-drawn "separate relations_2d topic" design.
        object_boxes = []   # (x1, y1, x2, y2, normalized_label)
        relation_hits = []  # (x1, y1, x2, y2, score)

        for index, (box, score, label) in enumerate(
            zip(
                result.boxes_xyxy,
                result.scores,
                result.labels,
            )
        ):
            x1, y1, x2, y2 = [
                float(value) for value in box
            ]

            x1 = max(0.0, min(x1, frame.shape[1] - 1.0))
            x2 = max(0.0, min(x2, frame.shape[1] - 1.0))
            y1 = max(0.0, min(y1, frame.shape[0] - 1.0))
            y2 = max(0.0, min(y2, frame.shape[0] - 1.0))

            if x2 <= x1 or y2 <= y1:
                continue

            normalized = normalize_phrase(label)
            is_relation = self.relation_phrases and any(
                rp in normalized or normalized in rp
                for rp in self.relation_phrases
            )

            if is_relation:
                relation_hits.append((x1, y1, x2, y2, float(score)))
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 255),
                    1,
                )
                continue

            detection = Detection2D()
            detection.header = msg.header
            detection.id = str(index)

            detection.bbox.center.position.x = (
                x1 + x2
            ) / 2.0
            detection.bbox.center.position.y = (
                y1 + y2
            ) / 2.0
            detection.bbox.center.theta = 0.0

            detection.bbox.size_x = x2 - x1
            detection.bbox.size_y = y2 - y1

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(label)
            hypothesis.hypothesis.score = float(score)

            detection.results.append(hypothesis)
            detection_array.detections.append(detection)
            object_boxes.append((x1, y1, x2, y2, normalize_phrase(label)))

            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )

            text = f"{label}: {float(score):.2f}"

            cv2.putText(
                annotated,
                text,
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Pass 2: match each relation box against this frame's own object
        # boxes (index-aligned with object_boxes / detection_array.detections,
        # appended together above) and tag the winning object. Pure geometry,
        # unit-tested separately -- see relation_matching.py.
        best_relconf = match_relations_to_objects(
            relation_hits, object_boxes, self.relation_iou_threshold)

        for obj_index, relconf in best_relconf.items():
            h = detection_array.detections[obj_index].results[0].hypothesis
            h.class_id = f"{h.class_id}|relconf={relconf:.2f}"

        self.detection_publisher.publish(detection_array)

        try:
            output_msg = self.bridge.cv2_to_imgmsg(
                annotated,
                encoding="bgr8",
            )
            output_msg.header = msg.header
            self.image_publisher.publish(output_msg)

        except CvBridgeError as error:
            self.get_logger().error(
                f"Annotated-image conversion failed: {error}"
            )

        if self.inference_count % 20 == 0:
            self.get_logger().info(
                f"Detections: "
                f"{len(detection_array.detections)}, "
                f"inference: {inference_time * 1000.0:.1f} ms"
            )


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = GroundingDinoDetectorNode()

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
