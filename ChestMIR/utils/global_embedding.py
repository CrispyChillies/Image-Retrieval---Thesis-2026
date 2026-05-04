"""Reusable ONNX global embedding utility for medical images.

This module provides lightweight helpers to:
- create/load ONNX Runtime sessions with CPU/CUDA fallback
- preprocess an image from path or from a BGR ndarray
- run embedding inference
- L2-normalize embeddings
"""

from __future__ import annotations

import argparse
import ctypes
import os
import site
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

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
import onnxruntime as ort


CUR_PATH = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = (
    CUR_PATH
    / "weights"
    / "retrieval_model"
    / "vindr_densenet121_seed_0_best_backbone.onnx"
)


PathLike = Union[str, Path]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_INPUT_SIZE = (224, 224)


def _prepare_cuda_library_path() -> None:
    """Prepend NVIDIA wheel library dirs so ORT uses env-matched CUDA libs."""
    _bootstrap_cuda_libs()


def _parse_hw_from_shape(shape: Sequence[object]) -> Optional[Tuple[int, int]]:
    """Extract static `(height, width)` from a model input shape if available."""
    if len(shape) != 4:
        return None
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        return int(h), int(w)
    return None


def _to_3ch_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 BGR with 3 channels."""
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 1:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        return image_bgr
    raise ValueError(f"Unsupported image shape: {image_bgr.shape}")


@lru_cache(maxsize=8)
def create_session(model_path: PathLike = DEFAULT_MODEL_PATH, device: str = "") -> ort.InferenceSession:
    """Create and cache an ONNX Runtime session with CUDA->CPU fallback.

    Args:
        model_path: Path to ONNX model.
        device: One of `""`, `"cpu"`, `"cuda"`, `"gpu"`.

    Returns:
        Initialized `onnxruntime.InferenceSession`.
    """
    model_path = str(Path(model_path))
    available = set(ort.get_available_providers())
    device_l = device.lower().strip()

    if device_l == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device_l in {"cuda", "gpu"}:
        _prepare_cuda_library_path()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. "
                "Install GPU-enabled onnxruntime and CUDA libraries."
            )
        # Strict CUDA path (no CPU fallback) to guarantee GPU inference.
        providers = ["CUDAExecutionProvider"]
    elif device_l in {"", "auto"}:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
    else:
        raise ValueError("Unsupported device. Use '', 'auto', 'cpu', or 'cuda'.")

    session = ort.InferenceSession(model_path, providers=providers)
    input_hw = _parse_hw_from_shape(session.get_inputs()[0].shape)
    is_default_retrieval_model = Path(model_path).name == Path(DEFAULT_MODEL_PATH).name
    allow_size_mismatch = os.environ.get("CHESTMIR_ALLOW_ONNX_INPUT_MISMATCH", "").strip() == "1"
    if is_default_retrieval_model and input_hw is not None and input_hw != DEFAULT_INPUT_SIZE:
        msg = (
            f"Unexpected ONNX input size {input_hw} for {Path(model_path).name}. "
            f"Expected {DEFAULT_INPUT_SIZE} to match test.py ConvNeXtV2 pipeline. "
            "This usually causes significant retrieval metric drop."
        )
        if allow_size_mismatch:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        else:
            raise RuntimeError(
                msg
                + " Re-export the retrieval ONNX with 384x384 input, or set "
                "CHESTMIR_ALLOW_ONNX_INPUT_MISMATCH=1 to bypass this check."
            )
    if device_l in {"cuda", "gpu"}:
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "Requested CUDA execution, but ONNX Runtime is not using CUDAExecutionProvider."
            )
        session.disable_fallback()
    return session


def preprocess_bgr_image(
    image_bgr: np.ndarray,
    session: Optional[ort.InferenceSession] = None,
    input_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Preprocess a BGR ndarray to model-ready NCHW float32 input tensor.

    Steps:
    - ensure 3-channel BGR
    - resize to model input size if inferable (or provided)
    - convert BGR->RGB
    - scale to `[0, 1]`
    - transpose to NCHW and add batch dimension

    Args:
        image_bgr: Input image in BGR format.
        session: Optional ONNX session used to infer input spatial shape.
        input_size: Optional `(height, width)` override.

    Returns:
        Preprocessed tensor with shape `(1, C, H, W)` and dtype `float32`.
    """
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy.ndarray")

    image_bgr = _to_3ch_bgr(image_bgr)
    target_size = input_size

    if target_size is None and session is not None:
        model_input_shape = session.get_inputs()[0].shape
        target_size = _parse_hw_from_shape(model_input_shape)
    if target_size is None:
        target_size = DEFAULT_INPUT_SIZE

    if target_size is not None:
        h, w = target_size
        if image_bgr.shape[0] != h or image_bgr.shape[1] != w:
            image_bgr = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32) / 255.0
    image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.transpose(image_float, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
    return np.ascontiguousarray(tensor)


def preprocess_image_path(
    image_path: PathLike,
    session: Optional[ort.InferenceSession] = None,
    input_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Load and preprocess an image from path to NCHW float32 tensor."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return preprocess_bgr_image(image, session=session, input_size=input_size)


def l2_normalize(embedding: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 1D embedding and return float32 output."""
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(emb)
    if norm < eps:
        return emb
    return (emb / norm).astype(np.float32, copy=False)


def infer_embedding(
    image_tensor: np.ndarray,
    session: ort.InferenceSession,
    normalize: bool = True,
) -> np.ndarray:
    """Run ONNX inference and return a float32 1D embedding vector.

    Args:
        image_tensor: Preprocessed tensor in NCHW float32 format.
        session: ONNX Runtime session.
        normalize: Whether to L2-normalize the output embedding.

    Returns:
        Embedding as `np.ndarray` (`float32`, 1D).
    """
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    output = session.run(None, {input_name: image_tensor})[0]

    emb = np.asarray(output, dtype=np.float32)
    if emb.ndim == 0:
        emb = emb.reshape(1)
    elif emb.ndim >= 2:
        emb = emb[0].reshape(-1)
    else:
        emb = emb.reshape(-1)

    if normalize:
        emb = l2_normalize(emb)
    return emb.astype(np.float32, copy=False)


def embed_from_bgr(
    image_bgr: np.ndarray,
    session: ort.InferenceSession,
    normalize: bool = True,
    input_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Convenience wrapper: preprocess BGR ndarray then infer embedding."""
    image_tensor = preprocess_bgr_image(image_bgr, session=session, input_size=input_size)
    return infer_embedding(image_tensor, session=session, normalize=normalize)


def embed_from_path(
    image_path: PathLike,
    session: ort.InferenceSession,
    normalize: bool = True,
    input_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Convenience wrapper: load image path, preprocess, then infer embedding."""
    image_tensor = preprocess_image_path(image_path, session=session, input_size=input_size)
    return infer_embedding(image_tensor, session=session, normalize=normalize)


def compute_global_embedding(
    image_path: PathLike,
    model_path: PathLike = DEFAULT_MODEL_PATH,
    device: str = "",
    normalize: bool = True,
    input_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """High-level API: create/load session and embed one image path."""
    session = create_session(model_path=model_path, device=device)
    return embed_from_path(image_path, session=session, normalize=normalize, input_size=input_size)


def get_global_embedding(
    image_path: PathLike,
    model_path: PathLike = DEFAULT_MODEL_PATH,
    device: str = "",
) -> np.ndarray:
    """Alias for compatibility with ingestion callers."""
    return compute_global_embedding(image_path=image_path, model_path=model_path, device=device)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ONNX global embedding smoke helper")
    parser.add_argument("--image", required=True, type=str, help="Path to image file")
    parser.add_argument(
        "--device",
        default="",
        type=str,
        help="Inference device: '', 'auto', 'cpu', or 'cuda'",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        type=str,
        help="Path to ONNX embedding model",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    sess = create_session(model_path=args.model_path, device=args.device)
    embedding = embed_from_path(args.image, session=sess, normalize=True)
    print(f"embedding_shape={embedding.shape}")
    print(f"embedding_dtype={embedding.dtype}")
    print(f"embedding_l2_norm={float(np.linalg.norm(embedding)):.6f}")
