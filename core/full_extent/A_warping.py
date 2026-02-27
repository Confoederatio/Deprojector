import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────
#  ASPECT RATIO NORMALIZATION
# ─────────────────────────────────────────────────────────────────────


def detect_sea_color(img):
    """Most frequent colour in a downsampled version of the image."""
    small = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    colors, counts = np.unique(
        small.reshape(-1, 3), axis=0, return_counts=True
    )
    return colors[np.argmax(counts)]


def pad_to_aspect(img, target_ar, fill_color):
    """
    Pad `img` (H, W, 3) with `fill_color` so its aspect ratio matches
    `target_ar` (= W / H).  Padding is added symmetrically.
    """
    h, w = img.shape[:2]
    current_ar = w / h

    if abs(current_ar - target_ar) < 1e-4:
        return img.copy(), {
            "top": 0,
            "bottom": 0,
            "left": 0,
            "right": 0,
        }

    if current_ar < target_ar:
        new_w = int(round(h * target_ar))
        pad_total = new_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        canvas = np.full((h, new_w, 3), fill_color, dtype=img.dtype)
        canvas[:, pad_left : pad_left + w, :] = img
        info = {
            "top": 0,
            "bottom": 0,
            "left": pad_left,
            "right": pad_right,
        }
    else:
        new_h = int(round(w / target_ar))
        pad_total = new_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        canvas = np.full((new_h, w, 3), fill_color, dtype=img.dtype)
        canvas[pad_top : pad_top + h, :, :] = img
        info = {
            "top": pad_top,
            "bottom": pad_bottom,
            "left": 0,
            "right": 0,
        }

    return canvas, info


def normalize_aspect_ratios(ref_bgr, src_bgr):
    sea_ref = detect_sea_color(ref_bgr)
    sea_src = detect_sea_color(src_bgr)
    ar_ref = ref_bgr.shape[1] / ref_bgr.shape[0]
    ar_src = src_bgr.shape[1] / src_bgr.shape[0]
    target_ar = ar_ref

    ref_padded, ref_pad = pad_to_aspect(ref_bgr, target_ar, sea_ref)
    src_padded, src_pad = pad_to_aspect(src_bgr, target_ar, sea_src)

    print(f"  Reference AR: {ar_ref:.4f}  →  no padding needed")
    print(
        f"  Source    AR: {ar_src:.4f}  →  padded "
        f"L={src_pad['left']} R={src_pad['right']} "
        f"T={src_pad['top']} B={src_pad['bottom']}"
    )
    print(
        f"  Reference padded size: "
        f"{ref_padded.shape[1]}×{ref_padded.shape[0]}"
    )
    print(
        f"  Source    padded size: "
        f"{src_padded.shape[1]}×{src_padded.shape[0]}"
    )
    return ref_padded, src_padded, ref_pad, src_pad


# ─────────────────────────────────────────────────────────────────────
#  MASK EXTRACTION
# ─────────────────────────────────────────────────────────────────────


def get_solid_mask(img):
    small = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    colors, counts = np.unique(
        small.reshape(-1, 3), axis=0, return_counts=True
    )
    sea_color = colors[np.argmax(counts)]
    diff = np.linalg.norm(
        img.astype(int) - sea_color.astype(int), axis=2
    )
    mask = np.where(diff > 35, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, contours, -1, 255, -1)
    return solid


# ─────────────────────────────────────────────────────────────────────
#  GAUSSIAN SMOOTHING UTILITY
# ─────────────────────────────────────────────────────────────────────


def gaussian_blur_2d(tensor, sigma):
    """
    Apply separable Gaussian blur to a (1, C, H, W) tensor.
    Used as a projection step to enforce spatial smoothness.
    """
    if sigma < 0.1:
        return tensor

    H, W = tensor.shape[2], tensor.shape[3]
    ks = max(3, int(6 * sigma) | 1)  # ensure odd

    # Clamp kernel to not exceed spatial dims (must be odd)
    max_ks = min(H, W)
    if max_ks < 3:
        return tensor
    if ks > max_ks:
        ks = max_ks if max_ks % 2 == 1 else max_ks - 1
    if ks < 3:
        return tensor

    ax = torch.arange(
        -(ks // 2),
        ks // 2 + 1,
        dtype=tensor.dtype,
        device=tensor.device,
        )
    kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    C = tensor.shape[1]
    kh = kernel_1d.view(1, 1, 1, -1).expand(C, -1, -1, -1)
    kv = kernel_1d.view(1, 1, -1, 1).expand(C, -1, -1, -1)
    pad = ks // 2

    out = F.conv2d(
        F.pad(tensor, [pad, pad, 0, 0], mode="reflect"),
        kh,
        groups=C,
    )
    out = F.conv2d(
        F.pad(out, [0, 0, pad, pad], mode="reflect"),
        kv,
        groups=C,
    )
    return out


# ─────────────────────────────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────


def dice_loss(pred, target, smooth=1.0):
    p = pred.view(-1)
    t = target.view(-1)
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + smooth) / (
            p.sum() + t.sum() + smooth
    )


def bending_energy_loss(disp, weight_map=None):
    """
    Second-order smoothness: penalise curvature (Laplacian) of the
    displacement field.  Enforces C¹ continuity → clinal (gradually
    varying) displacement vectors.

    Unlike first-order TV which only constrains displacement magnitude
    changes, this constrains the *rate of change* of the gradient,
    preventing abrupt directional shifts that cause mesh goop.

    disp:       (1, 2, mh, mw)
    weight_map: (1, 1, mh, mw)  –  high = strong smoothing
    """
    # ∂²f/∂x²
    d2x = (
            disp[:, :, :, 2:]
            - 2 * disp[:, :, :, 1:-1]
            + disp[:, :, :, :-2]
    )
    # ∂²f/∂y²
    d2y = (
            disp[:, :, 2:, :]
            - 2 * disp[:, :, 1:-1, :]
            + disp[:, :, :-2, :]
    )
    # ∂²f/∂x∂y  (cross-derivative)
    dxy = (
            disp[:, :, 1:, 1:]
            - disp[:, :, 1:, :-1]
            - disp[:, :, :-1, 1:]
            + disp[:, :, :-1, :-1]
    )

    if weight_map is not None:
        wx = weight_map[:, :, :, 1:-1]
        wy = weight_map[:, :, 1:-1, :]
        wxy = (
                      weight_map[:, :, 1:, 1:]
                      + weight_map[:, :, 1:, :-1]
                      + weight_map[:, :, :-1, 1:]
                      + weight_map[:, :, :-1, :-1]
              ) / 4.0
        return (
                (wx * d2x.pow(2)).mean()
                + (wy * d2y.pow(2)).mean()
                + 2.0 * (wxy * dxy.pow(2)).mean()
        )

    return (
            d2x.pow(2).mean()
            + d2y.pow(2).mean()
            + 2.0 * dxy.pow(2).mean()
    )


def fold_loss(grid):
    """Penalise mesh folding (negative Jacobian determinant)."""
    dg_dc_x = grid[:, :, 1:, 0] - grid[:, :, :-1, 0]
    dg_dc_y = grid[:, :, 1:, 1] - grid[:, :, :-1, 1]
    dg_dr_x = grid[:, 1:, :, 0] - grid[:, :-1, :, 0]
    dg_dr_y = grid[:, 1:, :, 1] - grid[:, :-1, :, 1]
    dg_dc_x = dg_dc_x[:, :-1, :]
    dg_dc_y = dg_dc_y[:, :-1, :]
    dg_dr_x = dg_dr_x[:, :, :-1]
    dg_dr_y = dg_dr_y[:, :, :-1]
    det = dg_dc_x * dg_dr_y - dg_dc_y * dg_dr_x
    return F.relu(-det + 0.05).mean()


def jacobian_regularity_loss(grid):
    """
    Penalise non-uniform Jacobian determinant across the mesh.
    Encourages locally uniform area scaling → prevents some cells
    from compressing to nothing while neighbours explode.
    """
    dg_dc_x = grid[:, :, 1:, 0] - grid[:, :, :-1, 0]
    dg_dc_y = grid[:, :, 1:, 1] - grid[:, :, :-1, 1]
    dg_dr_x = grid[:, 1:, :, 0] - grid[:, :-1, :, 0]
    dg_dr_y = grid[:, 1:, :, 1] - grid[:, :-1, :, 1]

    dg_dc_x = dg_dc_x[:, :-1, :]
    dg_dc_y = dg_dc_y[:, :-1, :]
    dg_dr_x = dg_dr_x[:, :, :-1]
    dg_dr_y = dg_dr_y[:, :, :-1]

    det = dg_dc_x * dg_dr_y - dg_dc_y * dg_dr_x
    mean_det = det.mean().detach()  # detach target to avoid collapse
    return (det - mean_det).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────
#  WARP HELPER
# ─────────────────────────────────────────────────────────────────────


def make_identity_grid(h, w, device):
    yy = torch.linspace(-1, 1, h, device=device)
    xx = torch.linspace(-1, 1, w, device=device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack([gx, gy], dim=-1).unsqueeze(0)


def warp(src, disp, out_h, out_w):
    device = src.device
    disp_full = F.interpolate(
        disp,
        size=(out_h, out_w),
        mode="bicubic",
        align_corners=True,
    )
    identity = make_identity_grid(out_h, out_w, device)
    grid = identity.clone()
    grid[..., 0] = grid[..., 0] + disp_full[:, 0]
    grid[..., 1] = grid[..., 1] + disp_full[:, 1]
    warped = F.grid_sample(
        src,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped, grid


# ─────────────────────────────────────────────────────────────────────
#  PER-CELL IoU & ADAPTIVE WEIGHT MAP
# ─────────────────────────────────────────────────────────────────────


def compute_cell_iou(warped_mask, ref_mask, mh, mw):
    _, _, H, W = warped_mask.shape
    cell_h = H // mh
    cell_w = W // mw
    iou_map = torch.full((mh, mw), -1.0)

    for i in range(mh):
        for j in range(mw):
            r0 = i * cell_h
            r1 = (i + 1) * cell_h if i < mh - 1 else H
            c0 = j * cell_w
            c1 = (j + 1) * cell_w if j < mw - 1 else W

            w_cell = warped_mask[0, 0, r0:r1, c0:c1] > 0.5
            r_cell = ref_mask[0, 0, r0:r1, c0:c1] > 0.5

            union = (w_cell | r_cell).sum().float()
            if union < 1.0:
                continue
            inter = (w_cell & r_cell).sum().float()
            iou_map[i, j] = (inter / union).item()

    return iou_map


def iou_to_bend_weights(cell_iou, high_w=5.0, low_w=0.1):
    """
    High IoU  → low bending weight (allow fine local adjustment)
    Low IoU   → high bending weight (stay smooth, don't thrash)
    No land   → high bending weight (don't distort empty ocean)
    """
    weights = torch.full_like(cell_iou, high_w)
    relevant = cell_iou >= 0.0
    weights[relevant] = high_w - (high_w - low_w) * cell_iou[relevant]
    return weights


# ─────────────────────────────────────────────────────────────────────
#  VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────


def show_mask_comparison(warped_mask, ref_mask, title=""):
    wm = warped_mask[0, 0].cpu().numpy()
    rm = ref_mask[0, 0].cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    axes[0].imshow(wm, cmap="gray")
    axes[0].set_title("Warped Source Mask")
    axes[1].imshow(rm, cmap="gray")
    axes[1].set_title("Reference Mask")
    rgb = np.zeros((*rm.shape, 3))
    rgb[..., 1] = (wm > 0.5) & (rm > 0.5)
    rgb[..., 0] = (rm > 0.5) & (wm <= 0.5)
    rgb[..., 2] = (wm > 0.5) & (rm <= 0.5)
    axes[2].imshow(rgb)
    axes[2].set_title("Overlap (G=hit R=miss B=extra)")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def show_mesh(grid_tensor, ref_img_np, step=8, title="Mesh"):
    g = grid_tensor[0].cpu().numpy()
    h, w = g.shape[:2]
    gx = (g[..., 0] + 1) / 2 * w
    gy = (g[..., 1] + 1) / 2 * h
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(ref_img_np)
    for r in range(0, h, step):
        ax.plot(gx[r, :], gy[r, :], "c-", lw=0.4, alpha=0.7)
    for c in range(0, w, step):
        ax.plot(gx[:, c], gy[:, c], "c-", lw=0.4, alpha=0.7)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_adaptive_maps(cell_iou, bend_weights, lvl, mh, mw):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iou_display = cell_iou.cpu().numpy().copy()
    iou_display[iou_display < 0] = np.nan
    im0 = axes[0].imshow(
        iou_display,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[0].set_title(
        f"Per-Cell IoU  (Level {lvl})\n"
        f"mesh {mw}×{mh}  |  grey = no land"
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(
        bend_weights.cpu().numpy(),
        cmap="hot",
        vmin=0,
        interpolation="nearest",
    )
    axes[1].set_title(
        "Adaptive Bending Weight → Next Level\n"
        "dark = fine (low bend)  |  bright = stiff (high bend)"
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes:
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def show_displacement_field(total_disp, lvl, mh, mw):
    """
    Visualise displacement magnitude, vectors, and gradient-of-magnitude.
    The third panel directly shows clinal continuity: if it's smooth
    (no sharp edges), the displacement field varies gradually.
    """
    d = total_disp[0].cpu().numpy()  # (2, mh, mw)
    mag = np.sqrt(d[0] ** 2 + d[1] ** 2)

    # Gradient of magnitude → clinal continuity indicator
    grad_y = np.gradient(mag, axis=0)
    grad_x = np.gradient(mag, axis=1)
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    im0 = axes[0].imshow(mag, cmap="magma", interpolation="bilinear")
    axes[0].set_title(
        f"Displacement Magnitude (Level {lvl})\n"
        f"mesh {mw}×{mh}"
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Quiver (subsample for readability)
    step_q = max(1, min(mh, mw) // 16)
    yy = np.arange(0, mh, step_q)
    xx = np.arange(0, mw, step_q)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    U = d[0][::step_q, ::step_q]
    V = d[1][::step_q, ::step_q]
    axes[1].imshow(mag, cmap="magma", interpolation="bilinear", alpha=0.4)
    axes[1].quiver(
        X, Y, U, V, mag[::step_q, ::step_q],
        cmap="coolwarm", scale=None, width=0.004,
    )
    axes[1].set_title("Displacement Vectors")

    im2 = axes[2].imshow(
        grad_mag, cmap="inferno", interpolation="bilinear"
    )
    axes[2].set_title(
        "‖∇ magnitude‖ (Clinal Continuity)\n"
        "Smooth = good  |  Sharp edges = bad"
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_ar_normalization(ref_orig, src_orig, ref_padded, src_padded):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes[0, 0].imshow(cv2.cvtColor(ref_orig, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(
        f"Reference original  "
        f"({ref_orig.shape[1]}×{ref_orig.shape[0]}  "
        f"AR={ref_orig.shape[1] / ref_orig.shape[0]:.3f})"
    )
    axes[0, 1].imshow(cv2.cvtColor(ref_padded, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(
        f"Reference padded  "
        f"({ref_padded.shape[1]}×{ref_padded.shape[0]})"
    )
    axes[1, 0].imshow(cv2.cvtColor(src_orig, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(
        f"Source original  "
        f"({src_orig.shape[1]}×{src_orig.shape[0]}  "
        f"AR={src_orig.shape[1] / src_orig.shape[0]:.3f})"
    )
    axes[1, 1].imshow(cv2.cvtColor(src_padded, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(
        f"Source padded  "
        f"({src_padded.shape[1]}×{src_padded.shape[0]})"
    )
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Aspect Ratio Normalization", fontsize=15)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────
#  MAIN: RESIDUAL COARSE-TO-FINE SGD MESH WARP WITH BENDING ENERGY
# ─────────────────────────────────────────────────────────────────────


def sgd_mesh_warp_adaptive(image_path_one, image_path_two):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}\n")

    # --- load ---------------------------------------------------------
    ref_bgr_orig = cv2.imread(image_path_one)
    src_bgr_orig = cv2.imread(image_path_two)
    if ref_bgr_orig is None:
        raise FileNotFoundError(f"Cannot read {image_path_one}")
    if src_bgr_orig is None:
        raise FileNotFoundError(f"Cannot read {image_path_two}")

    h_r_orig, w_r_orig = ref_bgr_orig.shape[:2]
    h_s_orig, w_s_orig = src_bgr_orig.shape[:2]
    print(f"Reference original : {w_r_orig}×{h_r_orig}")
    print(f"Source    original : {w_s_orig}×{h_s_orig}")

    # --- aspect ratio normalization -----------------------------------
    print("\n▸ Phase 0: Normalizing aspect ratios …")
    ref_bgr, src_bgr, ref_pad, src_pad = normalize_aspect_ratios(
        ref_bgr_orig, src_bgr_orig
    )
    show_ar_normalization(
        ref_bgr_orig, src_bgr_orig, ref_bgr, src_bgr
    )

    h_r, w_r = ref_bgr.shape[:2]
    h_s, w_s = src_bgr.shape[:2]
    print(f"\n  Reference padded : {w_r}×{h_r}")
    print(f"  Source    padded : {w_s}×{h_s}")
    print(f"  AR check: ref={w_r / h_r:.4f}  src={w_s / h_s:.4f}")

    # --- masks --------------------------------------------------------
    print("\n▸ Phase 1: Generating solid land masks …")
    mask_ref = get_solid_mask(ref_bgr)
    mask_src = get_solid_mask(src_bgr)

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mask_ref, cmap="gray")
    plt.title("Reference mask (AR-normalised)")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask_src, cmap="gray")
    plt.title("Source mask (AR-normalised)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- tensors ------------------------------------------------------
    ref_t = (
            torch.from_numpy(mask_ref)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
            / 255.0
    )
    src_mask_t = (
            torch.from_numpy(mask_src)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
            / 255.0
    )
    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    src_img_t = (
            torch.from_numpy(src_rgb)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            / 255.0
    )

    # --- working resolution -------------------------------------------
    COARSEST_CELL = 96
    work_h = 384
    raw_w = work_h * w_r / h_r
    work_w = max(
        COARSEST_CELL,
        int(round(raw_w / COARSEST_CELL)) * COARSEST_CELL,
        )
    print(f"\nWorking resolution: {work_w}×{work_h}")

    ref_sm = F.interpolate(
        ref_t, (work_h, work_w), mode="bilinear", align_corners=True
    )
    src_sm = F.interpolate(
        src_mask_t,
        (work_h, work_w),
        mode="bilinear",
        align_corners=True,
    )

    # --- coarse-to-fine schedule --------------------------------------
    cell_schedule = [96, 48, 24, 12, 6, 4, 2]
    steps_per_lvl = [600, 600, 600, 600, 600, 800, 800]
    lr_schedule = [6e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2e-4, 8e-5]
    lam_fold = 0.2
    lam_jac = 0.1

    # Bending energy: doubles each level. At cell=2 the mesh is
    # 192×(W/2) — without heavy bending penalty it WILL goop.
    lam_bend_schedule = [0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 50.0]

    # Residual magnitude: fine levels can only nudge, not rewrite
    lam_residual = [0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]

    # Projection sigma: still meaningful at fine levels since the
    # mesh cells are tiny — even sigma=0.2 spans real spatial extent
    proj_sigma_schedule = [2.0, 1.5, 1.0, 0.7, 0.5, 0.35, 0.2]

    # Project more frequently at finest levels
    proj_every = 50

    # Accumulated (frozen) base displacement – starts at zero
    base_disp = None
    bend_weight_map = None

    print("\n▸ Phase 2: Residual Coarse-to-Fine Optimisation …")
    print(
        f"  Cell sizes (px): {cell_schedule}  →  "
        f"finest mesh = "
        f"{work_w // cell_schedule[-1]}×"
        f"{work_h // cell_schedule[-1]}"
    )

    for lvl, (cell_sz, steps, lr) in enumerate(
            zip(cell_schedule, steps_per_lvl, lr_schedule)
    ):
        mh = work_h // cell_sz
        mw = work_w // cell_sz
        lam_bend = lam_bend_schedule[lvl]
        lam_res = lam_residual[lvl]
        proj_sigma = proj_sigma_schedule[lvl]

        print(
            f"\n  ── Level {lvl + 1}  cell={cell_sz}px  "
            f"mesh {mw}×{mh}  steps={steps}  lr={lr}  "
            f"λ_bend={lam_bend}  λ_res={lam_res}  "
            f"σ_proj={proj_sigma}"
        )

        # ---- upsample base from previous level ----------------------
        if base_disp is None:
            frozen_base = torch.zeros(
                1, 2, mh, mw, device=device
            )
        else:
            frozen_base = F.interpolate(
                base_disp.detach(),
                (mh, mw),
                mode="bicubic",
                align_corners=True,
            )
            # Smooth the upsampled base to prevent aliasing
            frozen_base = gaussian_blur_2d(
                frozen_base, sigma=1.0
            )

        # ---- residual: this is what we optimise ----------------------
        if lvl == 0:
            # First level: no prior, init from scratch
            residual = torch.zeros(
                1, 2, mh, mw, device=device, requires_grad=True
            )
        else:
            # Warm-start residual: difference between previous total
            # and new frozen base (accounts for smoothing)
            prev_total = F.interpolate(
                base_disp.detach(),
                (mh, mw),
                mode="bicubic",
                align_corners=True,
            )
            init_res = prev_total - frozen_base
            # Smooth the initial residual
            init_res = gaussian_blur_2d(
                init_res, sigma=proj_sigma
            )
            residual = init_res.clone().requires_grad_(True)

        # ---- build / upsample adaptive bending weight map -----------
        if bend_weight_map is None:
            bend_w = torch.ones(1, 1, mh, mw, device=device)
        else:
            bend_w = F.interpolate(
                bend_weight_map,
                (mh, mw),
                mode="bilinear",
                align_corners=True,
            )
        print(
            f"     Bending weights  min={bend_w.min().item():.2f}  "
            f"max={bend_w.max().item():.2f}  "
            f"mean={bend_w.mean().item():.2f}"
        )

        # ---- optimise ------------------------------------------------
        optim = torch.optim.Adam([residual], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, steps
        )

        best_loss = float("inf")
        best_residual = residual.data.clone()
        losses_log = []

        for step in range(steps):
            optim.zero_grad()

            # Total displacement = frozen base + learnable residual
            total_disp = frozen_base + residual
            warped, grid = warp(src_sm, total_disp, work_h, work_w)

            l_dice = dice_loss(warped, ref_sm)
            l_mse = F.mse_loss(warped, ref_sm)

            # Bending energy on TOTAL displacement (not just residual)
            # This ensures global smoothness, not just local
            l_bend = bending_energy_loss(total_disp, bend_w)

            l_fold = fold_loss(grid)
            l_jac = jacobian_regularity_loss(grid)

            # Residual magnitude penalty: prevent fine levels from
            # adding huge corrections that override the coarse solution
            l_res_mag = residual.pow(2).mean()

            loss = (
                    l_dice
                    + l_mse
                    + lam_bend * l_bend
                    + lam_fold * l_fold
                    + lam_jac * l_jac
                    + lam_res * l_res_mag
            )
            loss.backward()
            optim.step()
            sched.step()

            losses_log.append(loss.item())

            # ---- Gaussian projection: smooth the residual ------------
            if (step + 1) % proj_every == 0 and proj_sigma > 0.1:
                with torch.no_grad():
                    smoothed = gaussian_blur_2d(
                        residual.data, sigma=proj_sigma
                    )
                    residual.data.copy_(smoothed)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_residual = residual.data.clone()

            if step % 300 == 0 or step == steps - 1:
                with torch.no_grad():
                    td = frozen_base + residual
                    wc, _ = warp(src_sm, td, work_h, work_w)
                    hit = (
                        ((wc > 0.5) & (ref_sm > 0.5)).sum().item()
                    )
                    union = (
                        ((wc > 0.5) | (ref_sm > 0.5)).sum().item()
                    )
                    iou = hit / max(union, 1)
                print(
                    f"     step {step:5d}  "
                    f"loss={loss.item():.5f}  "
                    f"dice={l_dice.item():.4f}  "
                    f"bend={l_bend.item():.4f}  "
                    f"IoU={iou:.4f}"
                )

        # ---- finalise this level: merge into base --------------------
        best_total = frozen_base + best_residual
        # Final smooth of the merged total (light touch)
        base_disp = gaussian_blur_2d(
            best_total, sigma=proj_sigma * 0.5
        )

        # ---- per-cell IoU → adaptive bending weights for next level --
        with torch.no_grad():
            wv, _ = warp(src_sm, base_disp, work_h, work_w)
            cell_iou = compute_cell_iou(wv, ref_sm, mh, mw)
            raw_bw = iou_to_bend_weights(
                cell_iou, high_w=5.0, low_w=0.1
            )
            bend_weight_map = (
                raw_bw.unsqueeze(0).unsqueeze(0).to(device)
            )

        # ---- level visualisation ------------------------------------
        show_mask_comparison(
            wv,
            ref_sm,
            title=(
                f"After Level {lvl + 1}  "
                f"(cell {cell_sz}px → mesh {mw}×{mh})"
            ),
        )
        show_adaptive_maps(cell_iou, raw_bw, lvl + 1, mh, mw)
        show_displacement_field(base_disp, lvl + 1, mh, mw)

        plt.figure(figsize=(8, 3))
        plt.plot(losses_log, linewidth=0.8)
        plt.title(f"Level {lvl + 1} loss curve")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Use base_disp as the final displacement (already smoothed)
    current_disp = base_disp

    # ------------------------------------------------------------------
    #  Phase 3: Full-resolution warp (in padded coordinate space)
    # ------------------------------------------------------------------
    print("\n▸ Phase 3: Applying warp at full resolution …")
    with torch.no_grad():
        warped_color, final_grid = warp(
            src_img_t, current_disp, h_r, w_r
        )
        warped_mask_full, _ = warp(
            src_mask_t, current_disp, h_r, w_r
        )

    warped_np = (
            warped_color[0].permute(1, 2, 0).cpu().numpy() * 255
    ).astype(np.uint8)
    warped_bgr = cv2.cvtColor(warped_np, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    #  Phase 3b: Crop back to original reference frame
    # ------------------------------------------------------------------
    print("  Cropping to original reference frame …")
    t, b = ref_pad["top"], ref_pad["bottom"]
    l, r = ref_pad["left"], ref_pad["right"]
    crop_b = h_r - b if b > 0 else h_r
    crop_r = w_r - r if r > 0 else w_r
    warped_cropped = warped_bgr[t:crop_b, l:crop_r]
    ref_cropped = ref_bgr[t:crop_b, l:crop_r]

    print(
        f"  Cropped output: "
        f"{warped_cropped.shape[1]}×{warped_cropped.shape[0]}  "
        f"(matches original ref "
        f"{w_r_orig}×{h_r_orig})"
    )

    # ------------------------------------------------------------------
    #  Phase 4: Visualisation
    # ------------------------------------------------------------------
    print("\n▸ Phase 4: Visualisation …")
    ref_rgb = cv2.cvtColor(ref_cropped, cv2.COLOR_BGR2RGB)
    warped_rgb_cropped = cv2.cvtColor(
        warped_cropped, cv2.COLOR_BGR2RGB
    )

    show_mesh(
        final_grid,
        cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB),
        step=max(1, h_r // 80),
        title="Deformed Sampling Mesh (padded frame)",
    )

    plt.figure(figsize=(14, 7))
    plt.imshow(warped_rgb_cropped)
    plt.title("Warped Source Image (cropped to ref frame)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    overlay = cv2.addWeighted(
        ref_cropped, 0.4, warped_cropped, 0.6, 0
    )
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 7))
    plt.imshow(overlay_rgb)
    plt.title("Overlay (40 % ref + 60 % warped)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    show_mask_comparison(
        warped_mask_full, ref_t, title="Final Full-Resolution Overlap"
    )

    # Final displacement field visualization
    show_displacement_field(
        current_disp,
        "FINAL",
        current_disp.shape[2],
        current_disp.shape[3],
    )

    src_rgb_orig = cv2.cvtColor(src_bgr_orig, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    axes[0].imshow(src_rgb_orig)
    axes[0].set_title("Original Source")
    axes[1].imshow(warped_rgb_cropped)
    axes[1].set_title("Mesh-Warped Result")
    axes[2].imshow(ref_rgb)
    axes[2].set_title("Reference Target")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("\n✓ Done.")
    return warped_cropped, current_disp