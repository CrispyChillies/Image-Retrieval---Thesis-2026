from __future__ import annotations

import argparse
import ctypes
import inspect
import os
import site
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _bootstrap_cuda_libs() -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(":") if p]
    prepend: list[str] = []
    for sp in site.getsitepackages():
        nvidia_root = Path(sp) / "nvidia"
        if not nvidia_root.exists():
            continue
        for rel in (
            "cublas/lib",
            "cudnn/lib",
            "cuda_runtime/lib",
            "cufft/lib",
            "curand/lib",
            "cusolver/lib",
            "cusparse/lib",
            "nvjitlink/lib",
        ):
            p = str(nvidia_root / rel)
            if Path(p).exists() and p not in prepend:
                prepend.append(p)
    if prepend:
        merged = prepend + [p for p in existing_parts if p not in prepend]
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
    for sp in site.getsitepackages():
        nvidia_root = Path(sp) / "nvidia"
        if not nvidia_root.exists():
            continue
        for rel, soname in (
            ("cuda_runtime/lib", "libcudart.so.12"),
            ("cublas/lib", "libcublasLt.so.12"),
            ("cublas/lib", "libcublas.so.12"),
            ("cudnn/lib", "libcudnn.so.9"),
            ("curand/lib", "libcurand.so.10"),
            ("cufft/lib", "libcufft.so.11"),
            ("cusolver/lib", "libcusolver.so.11"),
            ("cusparse/lib", "libcusparse.so.12"),
            ("nvjitlink/lib", "libnvJitLink.so.12"),
        ):
            so = nvidia_root / rel / soname
            if so.exists():
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)


_bootstrap_cuda_libs()

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - surfaced only at runtime environments missing ORT
    ort = None  # type: ignore[assignment]
    _ORT_IMPORT_ERROR = exc
else:
    _ORT_IMPORT_ERROR = None


CHESTMIR_ROOT = Path(__file__).resolve().parents[1]
if str(CHESTMIR_ROOT) not in sys.path:
    sys.path.insert(0, str(CHESTMIR_ROOT))

from utils.inference import predict_image  # noqa: E402

try:
    from utils import global_embedding as _global_embedding  # type: ignore[attr-defined]  # noqa: E402
except Exception:
    _global_embedding = None


WEIGHTS_DIR = CHESTMIR_ROOT / "weights"
DEFAULT_REGION_EMBEDDING_ONNX = (
    WEIGHTS_DIR / "retrieval_model" / "covid_convnextv2_seed_0_epoch_16_backbone.onnx"
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_INPUT_SIZE = 384


def _get_model_input_hw(session: ort.InferenceSession) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    if len(shape) != 4:
        return DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        return h, w
    return DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE


def _run_global_helper_if_available(helper_name: str, *args: Any, **kwargs: Any) -> Any | None:
    if _global_embedding is None:
        return None

    helper = getattr(_global_embedding, helper_name, None)
    if not callable(helper):
        return None

    try:
        signature = inspect.signature(helper)
        if len(signature.parameters) == 0:
            return helper()
        return helper(*args, **kwargs)
    except Exception:
        return None


@lru_cache(maxsize=8)
def create_embedding_session(weights_path: str, device: str = "") -> ort.InferenceSession:
    """Create and cache an ONNXRuntime session for the region embedding model."""
    def _prepare_cuda_library_path() -> None:
        _bootstrap_cuda_libs()

    if ort is None:
        raise RuntimeError(
            "onnxruntime is required for region embedding but is not available"
        ) from _ORT_IMPORT_ERROR

    global_session = _run_global_helper_if_available("create_session", weights_path, device)
    if global_session is not None:
        return global_session

    available = ort.get_available_providers()
    if device.lower() == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device.lower() in {"cuda", "gpu"}:
        _prepare_cuda_library_path()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. "
                "Install GPU-enabled onnxruntime and CUDA libraries."
            )
        # Strict CUDA path (no CPU fallback) to guarantee GPU inference.
        providers = ["CUDAExecutionProvider"]
    elif device and device.lower() not in {"", "cpu", "cuda", "gpu"}:
        raise ValueError(f"Unsupported device: {device}. Use '', 'cpu', or 'cuda'.")
    else:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )

    session = ort.InferenceSession(weights_path, providers=providers)
    if device.lower() in {"cuda", "gpu"}:
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "Requested CUDA execution, but ONNX Runtime is not using CUDAExecutionProvider."
            )
        session.disable_fallback()
    return session


def preprocess_region_for_embedding(
    region_bgr: np.ndarray,
    input_hw: tuple[int, int],
) -> np.ndarray:
    """Convert a BGR crop into the ConvNeXtV2 ONNX input tensor."""
    helper_output = _run_global_helper_if_available("preprocess_image", region_bgr, input_hw)
    if isinstance(helper_output, np.ndarray):
        return helper_output.astype(np.float32, copy=False)

    h, w = input_hw
    rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return np.expand_dims(x, axis=0).astype(np.float32)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    helper_output = _run_global_helper_if_available("l2_normalize", vec)
    if isinstance(helper_output, np.ndarray):
        return helper_output.astype(np.float32, copy=False)

    denom = np.linalg.norm(vec) + 1e-12
    return (vec / denom).astype(np.float32, copy=False)


def embed_region(
    region_bgr: np.ndarray,
    *,
    session: ort.InferenceSession,
    normalize: bool = True,
) -> np.ndarray:
    """Embed one cropped lesion region and return a 1D float32 vector."""
    global_embed_vec = _run_global_helper_if_available(
        "embed_from_bgr",
        region_bgr,
        session=session,
        normalize=normalize,
    )
    if isinstance(global_embed_vec, np.ndarray):
        return global_embed_vec.astype(np.float32, copy=False)

    input_name = session.get_inputs()[0].name
    input_hw = _get_model_input_hw(session)
    tensor = preprocess_region_for_embedding(region_bgr, input_hw)
    outputs = session.run(None, {input_name: tensor})
    if not outputs:
        raise RuntimeError("Embedding ONNX model returned no outputs")

    vector = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    if normalize:
        vector = _l2_normalize(vector)
    return vector


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def serialize_region_embeddings(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert region embeddings to JSON-safe structures (vectors become Python lists)."""
    serialized: list[dict[str, Any]] = []
    for region in regions:
        item = dict(region)
        vector = item.get("region_vector")
        if isinstance(vector, np.ndarray):
            item["region_vector"] = vector.astype(np.float32).tolist()
        elif vector is None:
            item["region_vector"] = []
        else:
            item["region_vector"] = [float(v) for v in vector]
        serialized.append(item)
    return serialized


def compute_region_embeddings(
    image_bgr: np.ndarray,
    *,
    embedding_model_path: str | Path = DEFAULT_REGION_EMBEDDING_ONNX,
    stage: int = 2,
    img_size: int = 640,
    wbf_iou: float = 0.4,
    score_thres: float = 0.25,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """
    Run lesion detection and embed each detected region with ConvNeXtV2 ONNX.

    Returns one dict per region:
    - class_id: int
    - class_name_en: str
    - confidence: float
    - bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    - region_vector: np.ndarray[float32]
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or unreadable")

    embedding_model_path = Path(embedding_model_path)
    if not embedding_model_path.exists():
        raise FileNotFoundError(f"Missing embedding ONNX weights: {embedding_model_path}")

    detection_result = predict_image(
        image_bgr=image_bgr,
        stage=stage,
        img_size=img_size,
        wbf_iou=wbf_iou,
        score_thres=score_thres,
        device=device,
    )
    detections = detection_result.get("detections", [])
    if not detections:
        return []

    session = create_embedding_session(str(embedding_model_path), device)
    height, width = image_bgr.shape[:2]
    regions: list[dict[str, Any]] = []

    for detection in detections:
        x1 = int(detection["x1"])
        y1 = int(detection["y1"])
        x2 = int(detection["x2"])
        y2 = int(detection["y2"])
        x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        region_vector = embed_region(crop, session=session)
        regions.append(
            {
                "class_id": int(detection["class_id"]),
                "class_name_en": str(detection.get("class_name", detection.get("class_name_en", ""))),
                "confidence": float(detection["confidence"]),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "region_vector": region_vector,
            }
        )

    return regions


def compute_region_embeddings_from_path(
    image_path: str | Path,
    *,
    embedding_model_path: str | Path = DEFAULT_REGION_EMBEDDING_ONNX,
    stage: int = 2,
    img_size: int = 640,
    wbf_iou: float = 0.4,
    score_thres: float = 0.25,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Load an image from disk and compute per-region embeddings."""
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return compute_region_embeddings(
        image_bgr=image_bgr,
        embedding_model_path=embedding_model_path,
        stage=stage,
        img_size=img_size,
        wbf_iou=wbf_iou,
        score_thres=score_thres,
        device=device,
    )


def region_embeddings(image_path: str | Path, device: str = "cuda") -> list[dict[str, Any]]:
    """Alias helper for ingestion callers."""
    return compute_region_embeddings_from_path(image_path=image_path, device=device)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for lesion region embedding")
    parser.add_argument("image_path", type=str, help="Input image path")
    parser.add_argument(
        "--embedding-model-path",
        type=str,
        default=str(DEFAULT_REGION_EMBEDDING_ONNX),
        help="ConvNeXtV2 ONNX path",
    )
    parser.add_argument("--stage", type=int, default=2, help="Detection stage (1 or 2)")
    parser.add_argument("--img-size", type=int, default=640, help="Detection image size")
    parser.add_argument("--wbf-iou", type=float, default=0.4, help="WBF IoU threshold")
    parser.add_argument("--score-thres", type=float, default=0.25, help="Detection score threshold")
    parser.add_argument("--device", type=str, default="", help="''|cpu|cuda")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    region_outputs = compute_region_embeddings_from_path(
        image_path=args.image_path,
        embedding_model_path=args.embedding_model_path,
        stage=args.stage,
        img_size=args.img_size,
        wbf_iou=args.wbf_iou,
        score_thres=args.score_thres,
        device=args.device,
    )
    vector_dim = int(len(region_outputs[0]["region_vector"])) if region_outputs else 0
    print(f"regions={len(region_outputs)} vector_dim={vector_dim}")
