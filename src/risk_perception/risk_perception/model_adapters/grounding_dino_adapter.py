'''
gdino_adapter.py

- Load config and checkpoint once.
- Move he model to CUDA once.
- Accept a Numpy/OpenCV image.
- Run text-conditioned inference.
- Return:
	- boxes in pixel xyxy format
	- class labels
	- confidence scores.
'''

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from groundingdino.util.inference import Model
from groundingdino.util.inference import predict as gdino_predict

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    labels: List[str]


class GroundingDinoAdapter:
    """Model-only adapter with no ROS dependencies."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> None:
        self.device = device

        self.model = Model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=device,
        )

        # Diagnostics for the phrase-concatenation bug (WP-A#1). GDINO's
        # `Model.predict_with_caption()` calls the module-level `predict()`
        # with its library default `remove_combined=False`: every box's
        # phrase is then reconstructed from ALL caption tokens that score
        # above `text_threshold` anywhere in the image, not just the tokens
        # belonging to the " . "-separated prompt entry that box actually
        # matched. A two-object frame ("person . ground mobile robot .")
        # then comes back labelled "person ground mobile robot" for BOTH
        # boxes instead of one "person" and one "ground mobile robot". This
        # silently fragmented object_tracker_node's old exact-class_id
        # association (44 tracks for one Carter, 2026-09-08 -- see
        # docs/probes_2026-09.md) and is only partially masked by that
        # node's later `association_key: category` fix. `predict()` itself
        # (see inference.py) supports `remove_combined=True` to scope each
        # phrase to the single prompt entry between the nearest " . "/"[CLS]"
        # separators either side of the box's argmax token -- exactly what
        # this adapter wants -- but `predict_with_caption()` hardcodes the
        # old behaviour and takes no kwarg for it, so `predict()` below
        # calls the module-level `predict()` directly (replicating
        # `predict_with_caption()`'s few remaining lines: image
        # preprocessing + `Model.post_process_result()`) instead of going
        # through it. This counter tracks how many returned labels still
        # contain more than one prompt entry -- it should read 0 after the
        # fix; a nonzero, growing count would mean `remove_combined=True`
        # stopped taking effect (e.g. a groundingdino upgrade changed the
        # kwarg) without anything erroring.
        self._frame_count = 0
        self._multi_entry_label_count = 0
        self._window_multi_entry_label_count = 0

    def predict(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> DetectionResult:
        """
        Runs one GDINO inference pass and returns single-entry phrases.

        Equivalent to `Model.predict_with_caption()` except for the
        `remove_combined=True` kwarg passed to the module-level `predict()`
        -- see the docstring in `__init__` for why `predict_with_caption()`
        itself cannot be used. The return contract (boxes in pixel xyxy,
        scores, labels) is unchanged from before this fix.
        """
        processed_image = Model.preprocess_image(image_bgr=image_bgr).to(self.device)

        boxes, logits, phrases = gdino_predict(
            model=self.model.model,
            image=processed_image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
            remove_combined=True,
        )

        source_h, source_w = image_bgr.shape[:2]
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits,
        )

        boxes_xyxy = np.asarray(detections.xyxy, dtype=np.float32)

        if detections.confidence is None:
            scores = np.zeros((len(boxes_xyxy),), dtype=np.float32)
        else:
            scores = np.asarray(
                detections.confidence,
                dtype=np.float32,
            )

        # `remove_combined=True` can still hand back "" for a box whose
        # thresholded tokens don't reduce cleanly to one entry (see
        # `get_phrases_from_posmap`) -- map that to the sentinel "object"
        # rather than publishing an empty class_id downstream (the tracker
        # and risk_visualization.label_category both key off a non-empty
        # string).
        labels = [(str(phrase).strip() or "object") for phrase in phrases]

        self._count_multi_entry_labels(prompt, labels)

        return DetectionResult(
            boxes_xyxy=boxes_xyxy,
            scores=scores,
            labels=labels,
        )

    def _count_multi_entry_labels(self, prompt: str, labels: List[str]) -> None:
        """
        Counts, per frame, how many of this frame's labels contain MORE
        THAN ONE of the prompt's " . "-separated entries (e.g. the old bug's
        "person ground mobile robot"), and logs the running total once
        every 100 frames at INFO. Deliberately a substring test ("entry in
        label"), matching the same loose convention risk_visualization and
        relation_matching already use for phrase comparison in this
        codebase -- exact-token matching would need the tokenizer, which
        this adapter has no reason to expose.
        """
        entries = [entry.strip() for entry in prompt.split(" . ") if entry.strip()]

        multi_entry_labels = sum(
            1 for label in labels
            if sum(1 for entry in entries if entry and entry in label) > 1
        )

        self._frame_count += 1
        self._multi_entry_label_count += multi_entry_labels
        self._window_multi_entry_label_count += multi_entry_labels

        if self._frame_count % 100 == 0:
            LOGGER.info(
                "GroundingDinoAdapter phrase check: %d multi-entry label(s) "
                "over the last 100 frames (%d total since load, %d frames "
                "seen) -- should be 0 with remove_combined=True",
                self._window_multi_entry_label_count,
                self._multi_entry_label_count, self._frame_count,
            )
            self._window_multi_entry_label_count = 0
