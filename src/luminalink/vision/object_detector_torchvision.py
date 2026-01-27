from __future__ import annotations

from typing import Final

import numpy as np

from luminalink.types import DetectedObject


COCO_LABELS: Final[list[str]] = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class TorchVisionObjectDetector:
    """Object detection via torchvision's Faster R-CNN (optional dependency)."""

    def __init__(self, score_threshold: float = 0.5):
        self._score_threshold = score_threshold
        try:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.transforms.functional import to_tensor
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "TorchVision object detector requires extras: `uv sync --extra vision`"
            ) from e

        self._torch = torch
        self._to_tensor = to_tensor
        self._model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self._model.eval()

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._model.to(self._device)

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedObject]:
        """Run inference on a frame and return filtered detections."""

        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self._to_tensor(frame_rgb).to(self._device)
        with self._torch.inference_mode():
            out = self._model([x])[0]

        boxes = out["boxes"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        results: list[DetectedObject] = []
        for box, label, score in zip(boxes, labels, scores, strict=False):
            if float(score) < self._score_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            label_name = COCO_LABELS[int(label)] if int(label) < len(COCO_LABELS) else str(int(label))
            results.append(
                DetectedObject(
                    label=label_name,
                    confidence=float(score),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return results

