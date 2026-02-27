# ─────────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    warped_result, learned_disp, detected_bbox, final_iou = (
        sgd_mesh_warp_polygon_constrained(
            ref_img=ref_img,
            src_img=src_img,
            extent_result=result,
            work_h_bbox=384,
            cell_schedule=(48, 24, 12, 6, 4, 2),
            steps_per_lvl=(600, 600, 600, 600, 800, 800),
            lr_schedule=(5e-3, 3e-3, 1.5e-3, 8e-4, 3e-4, 1e-4),
            lam_bend_schedule=(0.5, 1.0, 2.0, 6.0, 15.0, 40.0),
            lam_residual=(0.0, 0.01, 0.03, 0.08, 0.2, 0.4),
            lam_fold=0.2,
            lam_jac=0.1,
            lam_leakage=5.0,
            proj_sigma_schedule=(2.0, 1.5, 1.0, 0.6, 0.35, 0.2),
            proj_every=50,
            max_disp_schedule=(0.6, 0.65, 0.7, 0.75, 0.8, 0.85),
            hard_clamp_every=10,
        )
    )

    cv2.imwrite("warped_polygon_constrained.png", warped_result)
    print(
        f"\nSaved warped_polygon_constrained.png  "
        f"(polygon IoU = {final_iou:.4f})"
    )

# ═══════════════════════════════════════════════════════════════════
#  11. RUN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from google.colab import drive
    # Import the preprocessor (assumes it's in the same directory
    # or notebook, e.g. saved as map_preprocessor.py)
    # from map_preprocessor import normalize_pair, visualize_segmentation, visualize_normalized_pair

    drive.mount("/content/drive")

    ref_path = "./drive/MyDrive/Colab Notebooks/landarea.png"
    src_path = "./drive/MyDrive/Colab Notebooks/image_03.png"

    ref_img = cv2.imread(ref_path)
    src_img = cv2.imread(src_path)

    assert ref_img is not None, f"Could not load {ref_path}"
    assert src_img is not None, f"Could not load {src_path}"

    print(f"Reference: {ref_img.shape}")
    print(f"Source:    {src_img.shape}\n")

    # ── Pre-process: segment and normalize ───────────────────────
    pair = normalize_pair(
        ref_img,
        src_img,
        work_dim=840,
        n_clusters=5,
        verbose=True,
    )

    # Visualize segmentation results
    visualize_segmentation(ref_img, pair.ref_seg, title="Reference")
    visualize_segmentation(src_img, pair.src_seg, title="Source")
    visualize_normalized_pair(pair)

    # ── Run extent finder with preprocessed data ─────────────────
    result = find_extent(
        ref_img,
        src_img,
        precomputed_pair=pair,
        mask_threshold=35,
        work_max_dim=840,
        loftr_dim=640,
        poly_degree=2,
        ransac_iters=5000,
        ransac_inlier_frac=0.025,
        use_anchor=True,
        verbose=True,
    )

    visualize_full(ref_img, src_img, result)
    visualize_correspondences(ref_img, src_img, result, n_show=80)
    visualize_ransac(ref_img, result)

    if result.warp_coeffs is not None:
        visualize_warp_grid(
            src_img.shape, ref_img, result.warp_coeffs
        )

    output = draw_extent(ref_img, result, thickness=4)
    cv2.imwrite("extent_result_v5.png", output)
    print("\nSaved extent_result_v5.png")