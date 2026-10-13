"""
ctypes bindings and automatic on-demand compiler for the C++17 photogrammetric engine.

This module compiles `src/ms_photogrammetry/cpp/photogrammetry_engine.cpp` into a native
shared library (`.dylib` on macOS, `.so` on Linux, `.dll` on Windows) on first import
and binds all 12 multithreaded C++17 photogrammetric kernels via `ctypes` using only
Python's standard library (`ctypes`, `array`, `os`, `shutil`, `subprocess`, `sys`).
"""

from __future__ import annotations
import array
import ctypes
import os
import shutil
import subprocess
import sys
from typing import Optional


_LIB_INSTANCE: Optional[ctypes.CDLL] = None


def _get_shared_lib_extension() -> str:
    """Return the platform-specific shared library extension."""
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform.startswith("win"):
        return ".dll"
    return ".so"


def _find_cpp_compiler() -> str:
    """Locate an available C++17 compiler (`CXX`, `clang++`, `g++`, or `c++`)."""
    env_cxx = os.environ.get("CXX")
    if env_cxx and shutil.which(env_cxx):
        return env_cxx
    for candidate in ("clang++", "g++", "c++"):
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError(
        "No C++17 compiler found. Please install clang++ or g++ (or set the CXX environment variable)."
    )


def get_native_lib() -> ctypes.CDLL:
    """
    Compile (if needed) and load the C++17 photogrammetric shared library.

    Automatically checks file modification timestamps between `photogrammetry_engine.cpp`
    and the compiled shared library so any C++ source edits trigger a transparent rebuild.

    Returns:
        ctypes.CDLL: Loaded shared library instance with configured `argtypes` and `restype`
        signatures for all 12 C++17 photogrammetric kernels.
    """
    global _LIB_INSTANCE
    if _LIB_INSTANCE is not None:
        return _LIB_INSTANCE

    cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp")
    src_path = os.path.join(cpp_dir, "photogrammetry_engine.cpp")
    ext = _get_shared_lib_extension()
    lib_path = os.path.join(cpp_dir, f"libphotogrammetry{ext}")

    needs_build = not os.path.exists(lib_path)
    if not needs_build and os.path.getmtime(src_path) > os.path.getmtime(lib_path):
        needs_build = True

    if needs_build:
        compiler = _find_cpp_compiler()
        cmd = [
            compiler,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
        ]
        if sys.platform.startswith("linux"):
            cmd.append("-pthread")
        cmd.extend([src_path, "-o", lib_path])
        subprocess.run(cmd, check=True)

    lib = ctypes.CDLL(lib_path)

    # 1. radiometric_calibrate_band
    lib.radiometric_calibrate_band.argtypes = [
        ctypes.c_void_p,                  # raw_dn (uint16_t*)
        ctypes.c_int,                     # width
        ctypes.c_int,                     # height
        ctypes.c_double,                  # black_level
        ctypes.c_void_p,                  # vig_exp_p (int*)
        ctypes.c_void_p,                  # vig_exp_q (int*)
        ctypes.c_void_p,                  # vig_coeffs (double*)
        ctypes.c_int,                     # num_vig_terms
        ctypes.c_double,                  # a1
        ctypes.c_double,                  # a2
        ctypes.c_double,                  # a3
        ctypes.c_double,                  # exposure_s
        ctypes.c_double,                  # gain
        ctypes.c_double,                  # temp_c
        ctypes.c_double,                  # radiance_to_refl
        ctypes.c_int,                     # is_thermal
        ctypes.c_void_p,                  # out_calibrated (float*)
    ]
    lib.radiometric_calibrate_band.restype = None

    # 2. undistort_points_2d
    lib.undistort_points_2d.argtypes = [
        ctypes.c_void_p,                  # uv_distorted (double*)
        ctypes.c_int,                     # num_points
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # fx, fy, cx, cy
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # k1..p2
        ctypes.c_void_p,                  # uv_undistorted (double*)
    ]
    lib.undistort_points_2d.restype = None

    # 3. undistort_image_float
    lib.undistort_image_float.argtypes = [
        ctypes.c_void_p,                  # src (float*)
        ctypes.c_int,                     # width
        ctypes.c_int,                     # height
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # fx, fy, cx, cy
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # k1..p2
        ctypes.c_void_p,                  # dst (float*)
    ]
    lib.undistort_image_float.restype = None

    # 4. detect_and_describe_features
    lib.detect_and_describe_features.argtypes = [
        ctypes.c_void_p,                  # img (float*)
        ctypes.c_int,                     # width
        ctypes.c_int,                     # height
        ctypes.c_int,                     # downsample_factor
        ctypes.c_int,                     # max_features
        ctypes.c_int,                     # grid_cols
        ctypes.c_int,                     # grid_rows
        ctypes.c_void_p,                  # out_keypoints_uv (double*)
        ctypes.c_void_p,                  # out_descriptors_64 (float*)
        ctypes.c_void_p,                  # out_scores (float*)
    ]
    lib.detect_and_describe_features.restype = ctypes.c_int

    # 5. match_descriptors_mutual
    lib.match_descriptors_mutual.argtypes = [
        ctypes.c_void_p,                  # kp1_uv (double*)
        ctypes.c_void_p,                  # desc1 (float*)
        ctypes.c_int,                     # n1
        ctypes.c_void_p,                  # kp2_uv (double*)
        ctypes.c_void_p,                  # desc2 (float*)
        ctypes.c_int,                     # n2
        ctypes.c_double,                  # prior_du
        ctypes.c_double,                  # prior_dv
        ctypes.c_double,                  # search_radius_px
        ctypes.c_double,                  # ratio_thresh
        ctypes.c_void_p,                  # out_match_idx1 (int*)
        ctypes.c_void_p,                  # out_match_idx2 (int*)
        ctypes.c_void_p,                  # out_match_dist (double*)
    ]
    lib.match_descriptors_mutual.restype = ctypes.c_int

    # 6. ransac_similarity_2d
    lib.ransac_similarity_2d.argtypes = [
        ctypes.c_void_p,                  # pts1_uv (double*)
        ctypes.c_void_p,                  # pts2_uv (double*)
        ctypes.c_int,                     # num_matches
        ctypes.c_double,                  # inlier_thresh_px
        ctypes.c_int,                     # max_iterations
        ctypes.c_void_p,                  # out_params_4 (double*)
        ctypes.c_void_p,                  # out_inlier_mask (int*)
        ctypes.c_void_p,                  # out_inlier_rmse (double*)
    ]
    lib.ransac_similarity_2d.restype = ctypes.c_int

    # 7. refine_band_offset_gradient_ncc
    lib.refine_band_offset_gradient_ncc.argtypes = [
        ctypes.c_void_p,                  # ref_band (float*)
        ctypes.c_void_p,                  # target_band (float*)
        ctypes.c_int,                     # width
        ctypes.c_int,                     # height
        ctypes.c_double,                  # init_du
        ctypes.c_double,                  # init_dv
        ctypes.c_double,                  # search_range_px
        ctypes.c_double,                  # step_px
        ctypes.c_void_p,                  # out_best_du (double*)
        ctypes.c_void_p,                  # out_best_dv (double*)
    ]
    lib.refine_band_offset_gradient_ncc.restype = ctypes.c_double

    # 8. accumulate_ortho_frame
    lib.accumulate_ortho_frame.argtypes = [
        ctypes.c_void_p,                  # raw_calibrated_band (float*)
        ctypes.c_int,                     # img_w
        ctypes.c_int,                     # img_h
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # fx, fy, cx, cy
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # k1..p2
        ctypes.c_void_p,                  # R_world_to_cam (double[9])
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # cam_easting, cam_northing, cam_alt
        ctypes.c_double, ctypes.c_double,  # band_du_px, band_dv_px
        ctypes.c_void_p,                  # dem_grid (float*)
        ctypes.c_int, ctypes.c_int,       # ortho_w, ortho_h
        ctypes.c_double, ctypes.c_double,  # ortho_origin_e, ortho_origin_n
        ctypes.c_double, ctypes.c_double,  # pixel_size_e, pixel_size_n
        ctypes.c_void_p,                  # ortho_accum_weighted (float*)
        ctypes.c_void_p,                  # ortho_accum_weight (float*)
        ctypes.c_void_p,                  # ortho_min_r2 (float*)
        ctypes.c_void_p,                  # ortho_obs_count (uint16_t*)
    ]
    lib.accumulate_ortho_frame.restype = None

    # 9. finalize_ortho_band
    lib.finalize_ortho_band.argtypes = [
        ctypes.c_void_p,                  # accum_weighted (float*)
        ctypes.c_void_p,                  # accum_weight (float*)
        ctypes.c_int,                     # total_pixels
        ctypes.c_double,                  # scale_to_uint16
        ctypes.c_double,                  # offset_before_scale
        ctypes.c_float,                   # nodata_float
        ctypes.c_void_p,                  # out_float32 (float*)
        ctypes.c_void_p,                  # out_uint16 (uint16_t*)
    ]
    lib.finalize_ortho_band.restype = ctypes.c_int

    # 10. compute_ndvi_and_rgb_previews
    lib.compute_ndvi_and_rgb_previews.argtypes = [
        ctypes.c_void_p,                  # blue (float*)
        ctypes.c_void_p,                  # green (float*)
        ctypes.c_void_p,                  # red (float*)
        ctypes.c_void_p,                  # nir (float*)
        ctypes.c_void_p,                  # red_edge (float*)
        ctypes.c_void_p,                  # lwir_c (float*)
        ctypes.c_void_p,                  # weight_mask (float*)
        ctypes.c_int,                     # width
        ctypes.c_int,                     # height
        ctypes.c_void_p,                  # out_ndvi (float*)
        ctypes.c_void_p,                  # out_ndre (float*)
        ctypes.c_void_p,                  # out_rgb_rgba (uint8_t*)
        ctypes.c_void_p,                  # out_cir_rgba (uint8_t*)
        ctypes.c_void_p,                  # out_ndvi_rgba (uint8_t*)
        ctypes.c_void_p,                  # out_thermal_rgba (uint8_t*)
    ]
    lib.compute_ndvi_and_rgb_previews.restype = None

    # 11. reconstruct_mvs_dense_dsm
    lib.reconstruct_mvs_dense_dsm.argtypes = [
        ctypes.c_void_p,                  # undist_images (float*)
        ctypes.c_int,                     # num_cams
        ctypes.c_int,                     # img_w
        ctypes.c_int,                     # img_h
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # fx, fy, cx, cy
        ctypes.c_void_p,                  # cam_poses (double*)
        ctypes.c_void_p,                  # prior_dem (float*)
        ctypes.c_void_p,                  # valid_mask (uint8_t*)
        ctypes.c_int,                     # grid_w
        ctypes.c_int,                     # grid_h
        ctypes.c_double,                  # origin_e
        ctypes.c_double,                  # origin_n
        ctypes.c_double,                  # cell_size_m
        ctypes.c_double,                  # z_search_half_range_m
        ctypes.c_int,                     # num_z_planes
        ctypes.c_void_p,                  # out_dsm (float*)
        ctypes.c_void_p,                  # out_zncc (float*)
        ctypes.c_void_p,                  # out_num_views (uint16_t*)
    ]
    lib.reconstruct_mvs_dense_dsm.restype = ctypes.c_int

    # 12. compute_dtm_chm_and_3d_renders
    lib.compute_dtm_chm_and_3d_renders.argtypes = [
        ctypes.c_void_p,                  # dsm_grid (float*)
        ctypes.c_void_p,                  # ndvi_grid (float*)
        ctypes.c_void_p,                  # rgb_grid_rgba (uint8_t*)
        ctypes.c_int,                     # grid_w
        ctypes.c_int,                     # grid_h
        ctypes.c_double,                  # cell_size_m
        ctypes.c_void_p,                  # out_dtm (float*)
        ctypes.c_void_p,                  # out_chm (float*)
        ctypes.c_void_p,                  # out_slope_deg (float*)
        ctypes.c_void_p,                  # out_rugosity (float*)
        ctypes.c_void_p,                  # out_normals (float*)
        ctypes.c_void_p,                  # out_class (uint8_t*)
        ctypes.c_void_p,                  # out_dsm_hillshade (uint8_t*)
        ctypes.c_void_p,                  # out_chm_rgba (uint8_t*)
        ctypes.c_int,                     # persp_w
        ctypes.c_int,                     # persp_h
        ctypes.c_void_p,                  # out_persp_rgba (uint8_t*)
    ]
    lib.compute_dtm_chm_and_3d_renders.restype = None

    _LIB_INSTANCE = lib
    return lib


def array_ptr(arr: array.array) -> int:
    """Return raw memory address of a Python standard library array.array."""
    return arr.buffer_info()[0]

