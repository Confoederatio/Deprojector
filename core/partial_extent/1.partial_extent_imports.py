# ═══════════════════════════════════════════════════════════════════
#  MAP EXTENT FINDER v5 — Monkey-Patched Surgical Implementation
# ═══════════════════════════════════════════════════════════════════

import sys
import types
from types import ModuleType

import cv2
import numpy as np
import torch
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")

if "local_corr" not in sys.modules:
    try:
        from romatch.utils import local_correlation as _lc

        # Create the fake module
        shim = types.ModuleType("local_corr")

        # Define a function that matches the signature RoMa expects,
        # but redirects to the native torch fallback.
        def local_corr_bridge(feat0, feat1, warp, mode='bilinear', normalized_coords=True):
            # We ignore some args because the native torch implementation
            # handles the grid sampling differently.
            B, C, H, W = feat0.shape
            # K is determined by the window size, we extract it from warp
            K = warp.shape[2]

            # Reconstruct the expected parameters for the native fallback
            # Note: RoMa's native fallback has a slightly different signature
            # We call the 'shitty' version which doesn't require the CUDA kernel.
            res, _ = _lc.shitty_native_torch_local_corr(
                feature0=feat0.permute(0, 2, 1).reshape(B, C, int(np.sqrt(H*W)), int(np.sqrt(H*W))),
                feature1=feat1.permute(0, 3, 1, 2),
                warp=warp[..., 0, :].reshape(B, int(np.sqrt(H*W)), int(np.sqrt(H*W)), 2),
                local_window=None, # Not needed by the internal logic if warp is full
                B=B, K=K, c=C, r=None, h=int(np.sqrt(H*W)), w=int(np.sqrt(H*W)),
                device=feat0.device,
                sample_mode=mode
            )
            return res

        # RoMa calls local_corr.local_corr(...)
        shim.local_corr = local_corr_bridge
        sys.modules["local_corr"] = shim

        # ALSO: Force RoMa to use the fallback path by monkey-patching
        # the wrapper to never even try the custom CUDA path.
        _lc.use_custom_corr = False

        print("  ✓ RoMa Bridge Active: Redirecting CUDA kernels to native PyTorch")
    except Exception as e:
        # Emergency backup: just an empty mock to prevent import errors
        mock = types.ModuleType("local_corr")
        mock.local_corr = lambda *args, **kwargs: torch.zeros(1)
        sys.modules["local_corr"] = mock
        print(f"  ! RoMa Bridge failed to initialize ({e}), using dummy mock")

# ── Optional imports ─────────────────────────────────────────────
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import kornia.feature as KF

    HAS_LOFTR = True
except ImportError:
    HAS_LOFTR = False

# We import both but will prioritize Tiny to avoid kernel errors.
HAS_ROMA_OUTDOOR = False
try:
    from romatch import roma_outdoor
    HAS_ROMA_OUTDOOR = True
except ImportError:
    pass

HAS_ROMA_TINY = False
try:
    try:
        from romatch import roma_tiny
    except ImportError:
        from romatch.models.model_zoo import roma_tiny
    HAS_ROMA_TINY = True
except ImportError:
    pass

HAS_ROMA = HAS_ROMA_OUTDOOR or HAS_ROMA_TINY