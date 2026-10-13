"""
Stage 2: Feature Detection, Descriptor Extraction, Brown-Conrady Undistortion,
and Geometric RANSAC Feature Matching across overlapping UAV captures.
"""

from __future__ import annotations
import array
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .native import get_native_lib, array_ptr
from .metadata import CaptureEvent
from .radiometry import RadiometricCalibrationModel, calibrate_band_image


@dataclass
class FrameFeatures:
    capture_index: int
    band_index: int
    num_keypoints: int
    keypoints_raw_uv: array.array       # double[2 * N] distorted pixel coords
    keypoints_undist_uv: array.array    # double[2 * N] undistorted pinhole pixel coords
    descriptors: array.array            # float[64 * N] L2-normalized gradient descriptors
    scores: array.array                 # float[N] Shi-Tomasi corner scores


@dataclass
class PairwiseMatch:
    idx1: int
    idx2: int
    pair_type: str                      # "along_track" or "cross_track"
    num_putative: int
    num_inliers: int
    inlier_ratio: float
    inlier_rmse_px: float
    similarity_params: Tuple[float, float, float, float]  # (a, b, tx, ty) mapping undist_uv1 -> undist_uv2
    inlier_kp_pairs: List[Tuple[int, int]] = field(default_factory=list)


def extract_features_for_capture(
    cap: CaptureEvent,
    band_index: int = 2,
    rad_model: Optional[RadiometricCalibrationModel] = None,
    downsample_factor: int = 2,
    max_features: int = 750,
    grid_cols: int = 10,
    grid_rows: int = 8,
) -> FrameFeatures:
    """Detect Shi-Tomasi corners, extract 64-D gradient descriptors, and undistort keypoint coordinates."""
    lib = get_native_lib()
    meta = cap.bands[band_index]
    cal_img = calibrate_band_image(meta, rad_model=rad_model)

    kp_raw = array.array("d", [0.0] * (max_features * 2))
    desc = array.array("f", [0.0] * (max_features * 64))
    scores = array.array("f", [0.0] * max_features)

    n_found = lib.detect_and_describe_features(
        array_ptr(cal_img),
        meta.width,
        meta.height,
        downsample_factor,
        max_features,
        grid_cols,
        grid_rows,
        array_ptr(kp_raw),
        array_ptr(desc),
        array_ptr(scores),
    )

    kp_raw = kp_raw[: 2 * n_found]
    desc = desc[: 64 * n_found]
    scores = scores[:n_found]
    kp_undist = array.array("d", [0.0] * (2 * n_found))

    if n_found > 0:
        k1, k2, k3, p1, p2 = meta.distortion_k123_p12
        lib.undistort_points_2d(
            array_ptr(kp_raw),
            n_found,
            meta.fx_px,
            meta.fy_px,
            meta.cx_px,
            meta.cy_px,
            k1,
            k2,
            k3,
            p1,
            p2,
            array_ptr(kp_undist),
        )

    return FrameFeatures(
        capture_index=cap.capture_index,
        band_index=band_index,
        num_keypoints=n_found,
        keypoints_raw_uv=kp_raw,
        keypoints_undist_uv=kp_undist,
        descriptors=desc,
        scores=scores,
    )


def match_feature_pair(
    f1: FrameFeatures,
    f2: FrameFeatures,
    pair_type: str = "along_track",
    prior_du: float = 0.0,
    prior_dv: float = 0.0,
    search_radius_px: float = 850.0,
    ratio_thresh: float = 0.88,
    ransac_thresh_px: float = 6.5,
    ransac_iters: int = 400,
) -> Optional[PairwiseMatch]:
    """Match two frames using mutual Lowe ratio test + 2D similarity RANSAC on undistorted keypoints."""
    if f1.num_keypoints < 8 or f2.num_keypoints < 8:
        return None

    lib = get_native_lib()
    max_m = min(f1.num_keypoints, f2.num_keypoints)
    idx1_arr = array.array("i", [0] * max_m)
    idx2_arr = array.array("i", [0] * max_m)
    dist_arr = array.array("d", [0.0] * max_m)

    n_putative = lib.match_descriptors_mutual(
        array_ptr(f1.keypoints_undist_uv),
        array_ptr(f1.descriptors),
        f1.num_keypoints,
        array_ptr(f2.keypoints_undist_uv),
        array_ptr(f2.descriptors),
        f2.num_keypoints,
        prior_du,
        prior_dv,
        search_radius_px,
        ratio_thresh,
        array_ptr(idx1_arr),
        array_ptr(idx2_arr),
        array_ptr(dist_arr),
    )

    if n_putative < 5:
        return None

    pts1 = array.array("d", [0.0] * (2 * n_putative))
    pts2 = array.array("d", [0.0] * (2 * n_putative))
    for k in range(n_putative):
        i1 = idx1_arr[k]
        i2 = idx2_arr[k]
        pts1[2 * k]     = f1.keypoints_undist_uv[2 * i1]
        pts1[2 * k + 1] = f1.keypoints_undist_uv[2 * i1 + 1]
        pts2[2 * k]     = f2.keypoints_undist_uv[2 * i2]
        pts2[2 * k + 1] = f2.keypoints_undist_uv[2 * i2 + 1]

    params_4 = array.array("d", [1.0, 0.0, 0.0, 0.0])
    inlier_mask = array.array("i", [0] * n_putative)
    rmse_out = array.array("d", [0.0])

    n_inliers = lib.ransac_similarity_2d(
        array_ptr(pts1),
        array_ptr(pts2),
        n_putative,
        ransac_thresh_px,
        ransac_iters,
        array_ptr(params_4),
        array_ptr(inlier_mask),
        array_ptr(rmse_out),
    )

    if n_inliers < 5:
        return None

    inlier_pairs: List[Tuple[int, int]] = []
    for k in range(n_putative):
        if inlier_mask[k] == 1:
            inlier_pairs.append((idx1_arr[k], idx2_arr[k]))

    return PairwiseMatch(
        idx1=f1.capture_index,
        idx2=f2.capture_index,
        pair_type=pair_type,
        num_putative=n_putative,
        num_inliers=n_inliers,
        inlier_ratio=n_inliers / max(1, n_putative),
        inlier_rmse_px=rmse_out[0],
        similarity_params=(params_4[0], params_4[1], params_4[2], params_4[3]),
        inlier_kp_pairs=inlier_pairs,
    )
