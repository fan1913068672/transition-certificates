from __future__ import annotations

from pathlib import Path
import torch


class GenericBarrierModel:
    """Minimal inference-only model for fc1->ReLU->fc2 barrier networks."""

    def __init__(self, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor):
        self.w1 = w1.float()
        self.b1 = b1.float()
        self.w2 = w2.float()
        self.b2 = b2.float()

    def __call__(self, *inputs):
        if len(inputs) != self.w1.shape[1]:
            raise ValueError(f"Expected {self.w1.shape[1]} inputs, got {len(inputs)}")
        x = torch.tensor([float(v) for v in inputs], dtype=torch.float32)  # (in_dim,)
        h = torch.relu(self.w1 @ x + self.b1)  # (hidden,)
        y = self.w2 @ h + self.b2  # (1,)
        return y.squeeze()


def _extract_state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported checkpoint format.")


def load_model(model_path: Path | str, input_dim: int | None = None):
    ckpt = torch.load(str(model_path), map_location="cpu")
    sd = _extract_state_dict(ckpt)

    # Support common key names used in this project.
    w1 = sd.get("fc1.weight")
    b1 = sd.get("fc1.bias")
    w2 = sd.get("fc2.weight")
    b2 = sd.get("fc2.bias")

    if any(v is None for v in (w1, b1, w2, b2)):
        raise ValueError("Checkpoint missing fc1/fc2 parameters.")
    if input_dim is not None and w1.shape[1] != input_dim:
        raise ValueError(f"Input dim mismatch: checkpoint={w1.shape[1]}, expected={input_dim}")

    return GenericBarrierModel(w1, b1, w2, b2)


def find_latest_model(root: Path | str) -> Path:
    root = Path(root)
    candidates = []

    # Prefer saved_models/barrier_net.pth produced by main.py
    candidates.extend(root.rglob("saved_models/**/barrier_net.pth"))
    # Fallback: any .pth in subtree
    candidates.extend(root.rglob("*.pth"))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No .pth model found under: {root}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

