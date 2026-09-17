from dataclasses import dataclass
from contextlib import nullcontext

import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass
class SegmentationResult:
    """SAM 2 segmentation result for one image."""

    masks: np.ndarray
    scores: np.ndarray


class Sam2Adapter:
    """Model-only SAM 2 adapter without ROS dependencies."""

    def __init__(
        self,
        model_config: str,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> None:
        self.device = device

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "SAM 2 requested CUDA, but torch.cuda.is_available() is False"
            )

        model = build_sam2(
            model_config,
            checkpoint_path,
            device=device,
        )

        self.predictor = SAM2ImagePredictor(model)

    def segment(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> SegmentationResult:
        """
        Segment one image using N box prompts.

        Parameters
        ----------
        image_rgb:
            H x W x 3 uint8 RGB image.
        boxes_xyxy:
            N x 4 float32 array of [x1, y1, x2, y2].

        Returns
        -------
        SegmentationResult:
            masks has shape N x H x W and Boolean dtype.
        """

        boxes_xyxy = np.asarray(
            boxes_xyxy,
            dtype=np.float32,
        )

        if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
            raise ValueError(
                f"Expected boxes with shape N x 4, got {boxes_xyxy.shape}"
            )

        if len(boxes_xyxy) == 0:
            height, width = image_rgb.shape[:2]

            return SegmentationResult(
                masks=np.zeros(
                    (0, height, width),
                    dtype=bool,
                ),
                scores=np.zeros((0,), dtype=np.float32),
            )

        amp_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
            if self.device == "cuda"
            else nullcontext()
        )

        with torch.inference_mode():
            with amp_context:
                self.predictor.set_image(image_rgb)

                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes_xyxy,
                    multimask_output=False,
                )

        masks = np.asarray(masks)
        scores = np.asarray(scores)

        # SAM 2 may return N x 1 x H x W when multimask_output=False.
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        elif masks.ndim == 2:
            masks = masks[None, :, :]
        elif masks.ndim != 3:
            raise RuntimeError(
                f"Unexpected SAM 2 mask shape: {masks.shape}"
            )

        scores = scores.reshape(-1)

        return SegmentationResult(
            masks=masks.astype(bool),
            scores=scores.astype(np.float32),
        )
