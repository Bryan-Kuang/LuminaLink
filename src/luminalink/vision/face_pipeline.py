from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """A face detection bounding box."""

    bbox_xyxy: tuple[int, int, int, int]


class FacePipeline:
    """Face detection + embedding extraction (best-effort with optional ML backends)."""

    def __init__(self):
        self._backend = self._select_backend()
        self.model_name = self._backend["model_name"]

    def detect_faces(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        """Detect faces in the frame."""

        return self._backend["detect_faces"](frame_bgr)

    def embed_faces(self, frame_bgr: np.ndarray, faces: list[FaceDetection]) -> list[np.ndarray]:
        """Compute embeddings for each detected face."""

        return self._backend["embed_faces"](frame_bgr, faces)

    def embedding_from_image_path(self, image_path: str) -> np.ndarray:
        """Extract a single face embedding from a still image."""

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        faces = self.detect_faces(img)
        if not faces:
            raise ValueError("No face detected")
        embeddings = self.embed_faces(img, [faces[0]])
        return embeddings[0]

    def _select_backend(self) -> dict:
        """Select the best available face pipeline backend."""

        try:
            from facenet_pytorch import InceptionResnetV1, MTCNN
            import torch
        except Exception:
            return self._opencv_backend()

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        mtcnn = MTCNN(keep_all=True, device=device)
        embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

        def detect_faces(frame_bgr: np.ndarray) -> list[FaceDetection]:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes, _probs = mtcnn.detect(frame_rgb)
            if boxes is None:
                return []
            out: list[FaceDetection] = []
            for x1, y1, x2, y2 in boxes.tolist():
                out.append(
                    FaceDetection(
                        bbox_xyxy=(int(x1), int(y1), int(x2), int(y2))
                    )
                )
            return out

        def embed_faces(frame_bgr: np.ndarray, faces: list[FaceDetection]) -> list[np.ndarray]:
            if not faces:
                return []
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes = np.array([list(f.bbox_xyxy) for f in faces], dtype=np.float32)
            aligned = mtcnn.extract(frame_rgb, boxes, save_path=None)
            if aligned is None:
                return []
            batch = aligned.to(device)
            with torch.inference_mode():
                emb = embedder(batch).detach().cpu().numpy()
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            return [emb[i] for i in range(emb.shape[0])]

        return {
            "model_name": "facenet_pytorch_vggface2",
            "detect_faces": detect_faces,
            "embed_faces": embed_faces,
        }

    def _opencv_backend(self) -> dict:
        """Fallback backend using OpenCV Haar cascade and color-histogram embeddings."""

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))

        def detect_faces(frame_bgr: np.ndarray) -> list[FaceDetection]:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            out: list[FaceDetection] = []
            for x, y, w, h in rects:
                out.append(FaceDetection(bbox_xyxy=(int(x), int(y), int(x + w), int(y + h))))
            return out

        def embed_faces(frame_bgr: np.ndarray, faces: list[FaceDetection]) -> list[np.ndarray]:
            embeddings: list[np.ndarray] = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox_xyxy
                crop = frame_bgr[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                if crop.size == 0:
                    continue
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
                hist = hist / (np.linalg.norm(hist) + 1e-9)
                embeddings.append(hist)
            return embeddings

        return {
            "model_name": "opencv_haar_colorhist",
            "detect_faces": detect_faces,
            "embed_faces": embed_faces,
        }
