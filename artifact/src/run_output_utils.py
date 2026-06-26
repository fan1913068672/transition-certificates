from __future__ import annotations

import time
from typing import Any


def now_ts() -> float:
    return time.time()


def print_header(example: str, method: str, cert_type: str, config: dict[str, Any] | None = None) -> None:
    print(f"[INFO] method={method} example={example} cert_type={cert_type}")
    if config:
        cfg = " ".join(f"{k}={v}" for k, v in config.items())
        print(f"[CONFIG] {cfg}")


def print_result(success: bool, iterations: int | None, elapsed_sec: float, out_path: str) -> None:
    print(
        f"[RESULT] success={str(success).lower()} "
        f"iterations={iterations if iterations is not None else 'na'} "
        f"elapsed_sec={elapsed_sec:.4f}"
    )
    print(f"[RESULT] output={out_path}")

