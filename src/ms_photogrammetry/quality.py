"""
Stage 8: Quality Assessment, Geometric & Radiometric Diagnostics, and Trajectory Plotting.
Implemented from scratch using Python's standard library.
"""

from __future__ import annotations
import json
import math
import os
from dataclasses import asdict
from typing import Dict, List, Any

from .metadata import DatasetInspectionReport, utm_to_wgs84
from .radiometry import RadiometricCalibrationModel
from .registration import MultispectralRegistrationModel
from .sfm import SfMResult
from .orthorectification import OrthomosaicProduct
from .tiff_io import read_geotiff_header_summary, write_png_rgb


def render_trajectory_and_tiepoints_png(
    sfm_result: SfMResult,
    ortho_product: OrthomosaicProduct,
    filepath: str,
    width: int = 1000,
    height: int = 560,
) -> None:
    """
    Render a diagnostic 2D map of triangulated 3D tie points, GPS priors vs
    Bundle-Adjusted camera centers, and pairwise RANSAC links as a PNG image.
    """
    buf = bytearray([18, 22, 31] * (width * height))  # sleek dark background

    min_e = ortho_product.origin_easting
    max_e = min_e + ortho_product.width * ortho_product.pixel_size_m
    max_n = ortho_product.origin_northing
    min_n = max_n - ortho_product.height * ortho_product.pixel_size_m

    span_e = max(1.0, max_e - min_e)
    span_n = max(1.0, max_n - min_n)
    pad = 40
    scale = min((width - 2 * pad) / span_e, (height - 2 * pad) / span_n)

    def to_px(e: float, n: float) -> Tuple[int, int]:
        x = int(pad + (e - min_e) * scale)
        y = int(pad + (max_n - n) * scale)
        return x, y

    def set_px(x: int, y: int, r: int, g: int, b: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            off = (y * width + x) * 3
            buf[off] = r
            buf[off + 1] = g
            buf[off + 2] = b

    def draw_disc(cx: int, cy: int, rad: int, r: int, g: int, b: int) -> None:
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                if dx * dx + dy * dy <= rad * rad:
                    set_px(cx + dx, cy + dy, r, g, b)

    def draw_line(x0: int, y0: int, x1: int, y1: int, r: int, g: int, b: int) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            t = s / steps
            x = int(round(x0 + t * (x1 - x0)))
            y = int(round(y0 + t * (y1 - y0)))
            set_px(x, y, r, g, b)

    # 1. Draw 3D tie points colored by reprojection accuracy (green = low error, amber = higher)
    for tp in sfm_result.tie_points:
        px, py = to_px(tp.easting, tp.northing)
        t = min(1.0, tp.mean_reproj_error_px / 6.0)
        r = int(40 + 180 * t)
        g = int(200 - 100 * t)
        b = int(120 - 80 * t)
        draw_disc(px, py, 1, r, g, b)

    # 2. Draw pairwise RANSAC connectivity graph
    for m in sfm_result.pairwise_matches:
        if m.idx1 in sfm_result.poses and m.idx2 in sfm_result.poses:
            p1 = sfm_result.poses[m.idx1]
            p2 = sfm_result.poses[m.idx2]
            x0, y0 = to_px(p1.easting, p1.northing)
            x1, y1 = to_px(p2.easting, p2.northing)
            if m.pair_type == "along_track":
                draw_line(x0, y0, x1, y1, 70, 140, 230)
            else:
                draw_line(x0, y0, x1, y1, 90, 95, 130)

    # 3. Draw raw GPS positions (red/orange) and Bundle-Adjusted camera centers (bright cyan)
    for p in sfm_result.poses.values():
        gx, gy = to_px(p.gps_easting, p.gps_northing)
        bx, by = to_px(p.easting, p.northing)
        draw_line(gx, gy, bx, by, 255, 110, 90)
        draw_disc(gx, gy, 2, 240, 90, 70)
        draw_disc(bx, by, 3, 60, 230, 255)

    write_png_rgb(filepath, bytes(buf), width, height)


def evaluate_pipeline_quality(
    inspection: DatasetInspectionReport,
    rad_model: RadiometricCalibrationModel,
    sfm_result: SfMResult,
    reg_model: MultispectralRegistrationModel,
    ortho_product: OrthomosaicProduct,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Stage 8: Compute and export comprehensive photogrammetric quality metrics,
    verify the written GeoTIFF headers, and render the trajectory diagnostic plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    traj_png = os.path.join(output_dir, "trajectory_and_tiepoints.png")
    render_trajectory_and_tiepoints_png(sfm_result, ortho_product, traj_png)

    # Verify GeoTIFF tags from disk
    geotiff_check = read_geotiff_header_summary(ortho_product.multiband_uint16_geotiff)

    # Compute corner WGS84 coordinates
    ul_lat, ul_lon = utm_to_wgs84(ortho_product.origin_easting, ortho_product.origin_northing, zone=14)
    lr_e = ortho_product.origin_easting + ortho_product.width * ortho_product.pixel_size_m
    lr_n = ortho_product.origin_northing - ortho_product.height * ortho_product.pixel_size_m
    lr_lat, lr_lon = utm_to_wgs84(lr_e, lr_n, zone=14)

    # Compute reprojection error percentiles across all 3D tie points
    tp_errors = sorted([tp.mean_reproj_error_px for tp in sfm_result.tie_points])
    p50_reproj = tp_errors[len(tp_errors) // 2] if tp_errors else 0.0
    p90_reproj = tp_errors[int(len(tp_errors) * 0.90)] if tp_errors else 0.0

    ground_reproj_rmse_cm = sfm_result.final_reproj_rmse_px * inspection.mean_gsd_ms_m * 100.0
    check_ground_rmse_cm = sfm_result.check_point_rmse_px * inspection.mean_gsd_ms_m * 100.0

    along_matches = [m for m in sfm_result.pairwise_matches if m.pair_type == "along_track"]
    cross_matches = [m for m in sfm_result.pairwise_matches if m.pair_type == "cross_track"]

    quality_summary: Dict[str, Any] = {
        "dataset": {
            "total_files": inspection.total_files,
            "total_captures": inspection.total_captures,
            "survey_captures_count": len(inspection.survey_captures),
            "calibration_panel_captures": inspection.calibration_captures,
            "excluded_transit_captures_count": len(inspection.transit_captures),
            "num_flight_lines": len(inspection.flight_lines),
            "ground_elevation_msl_m": round(inspection.ground_elevation_m, 3),
            "mean_survey_altitude_msl_m": round(inspection.mean_survey_altitude_msl_m, 3),
            "mean_agl_m": round(inspection.mean_agl_m, 3),
            "native_gsd_ms_cm": round(inspection.mean_gsd_ms_m * 100.0, 3),
            "native_gsd_panchro_cm": round(inspection.mean_gsd_panchro_m * 100.0, 3),
            "native_gsd_lwir_cm": round(inspection.mean_gsd_lwir_m * 100.0, 3),
            "single_frame_footprint_m": [round(inspection.footprint_ms_m[0], 2), round(inspection.footprint_ms_m[1], 2)],
            "mean_forward_spacing_m": round(inspection.mean_forward_step_m, 2),
            "mean_flight_line_spacing_m": round(inspection.mean_line_spacing_m, 2),
            "forward_overlap_pct": round(inspection.forward_overlap_pct, 2),
            "side_overlap_pct": round(inspection.side_overlap_pct, 2),
            "band_specifications": inspection.band_specs,
        },
        "radiometric_calibration": {
            "panel_dls_correction_factors": {str(k): round(v, 4) for k, v in rad_model.panel_dls_correction.items()},
            "panel_albedos": {str(k): round(v, 4) for k, v in rad_model.panel_albedos.items()},
            "panel_radiances_W_m2_sr_nm": {str(k): round(v, 5) for k, v in rad_model.panel_radiances.items()},
        },
        "sfm_and_bundle_adjustment": {
            "connected_survey_captures": sfm_result.connected_captures,
            "total_survey_captures": len(inspection.survey_captures),
            "verified_pairwise_links": len(sfm_result.pairwise_matches),
            "along_track_links": len(along_matches),
            "cross_track_links": len(cross_matches),
            "mean_inliers_along_track": round(sum(m.num_inliers for m in along_matches) / max(1, len(along_matches)), 1),
            "mean_inliers_cross_track": round(sum(m.num_inliers for m in cross_matches) / max(1, len(cross_matches)), 1),
            "triangulated_3d_tie_points": len(sfm_result.tie_points),
            "initial_reprojection_rmse_px": round(sfm_result.initial_reproj_rmse_px, 3),
            "final_reprojection_rmse_px": round(sfm_result.final_reproj_rmse_px, 3),
            "median_tie_point_reproj_error_px": round(p50_reproj, 3),
            "p90_tie_point_reproj_error_px": round(p90_reproj, 3),
            "independent_check_tie_point_rmse_px": round(sfm_result.check_point_rmse_px, 3),
            "equivalent_ground_reproj_rmse_cm": round(ground_reproj_rmse_cm, 2),
            "independent_check_ground_rmse_cm": round(check_ground_rmse_cm, 2),
            "mean_gps_xy_adjustment_m": round(sfm_result.mean_gps_XY_shift_m, 3),
            "mean_gps_z_adjustment_m": round(sfm_result.mean_gps_Z_shift_m, 3),
        },
        "multispectral_registration": {
            str(b): asdict(params) for b, params in sorted(reg_model.bands.items())
        },
        "orthomosaic_product": {
            "crs": "EPSG:32614 (WGS 84 / UTM zone 14N)",
            "dimensions_px": [ortho_product.width, ortho_product.height],
            "pixel_resolution_m": ortho_product.pixel_size_m,
            "pixel_resolution_cm": round(ortho_product.pixel_size_m * 100.0, 2),
            "utm_bounds_easting_northing": [
                round(ortho_product.origin_easting, 3),
                round(lr_n, 3),
                round(lr_e, 3),
                round(ortho_product.origin_northing, 3),
            ],
            "wgs84_bounds_lat_lon": {
                "upper_left": [round(ul_lat, 7), round(ul_lon, 7)],
                "lower_right": [round(lr_lat, 7), round(lr_lon, 7)],
            },
            "valid_pixels": ortho_product.valid_pixels,
            "total_grid_pixels": ortho_product.total_pixels,
            "valid_coverage_pct": round(100.0 * ortho_product.valid_pixels / max(1, ortho_product.total_pixels), 2),
            "geographic_coverage_area_m2": round(ortho_product.coverage_area_m2, 2),
            "mean_overlapping_images_per_pixel": round(ortho_product.mean_overlap_observations, 2),
            "dem_elevation_range_m": [
                round(ortho_product.dem_min_m, 3),
                round(ortho_product.dem_mean_m, 3),
                round(ortho_product.dem_max_m, 3),
            ],
            "band_statistics": ortho_product.band_stats,
            "geotiff_verification": {
                "magic": geotiff_check.get("magic"),
                "width": geotiff_check.get("width"),
                "height": geotiff_check.get("height"),
                "samples_per_pixel": geotiff_check.get("samples_per_pixel"),
                "planar_configuration": geotiff_check.get("planar_configuration"),
                "pixel_scale": geotiff_check.get("pixel_scale"),
                "tiepoint": geotiff_check.get("tiepoint"),
                "geo_ascii": geotiff_check.get("geo_ascii"),
                "gdal_nodata": geotiff_check.get("gdal_nodata"),
            },
            "files": {
                "multiband_uint16_geotiff": ortho_product.multiband_uint16_geotiff,
                "multiband_float32_geotiff": ortho_product.multiband_float32_geotiff,
                "ndvi_ndre_geotiff": ortho_product.ndvi_geotiff,
                "rgb_preview_png": ortho_product.rgb_preview_png,
                "cir_preview_png": ortho_product.cir_preview_png,
                "ndvi_preview_png": ortho_product.ndvi_preview_png,
                "thermal_preview_png": ortho_product.thermal_preview_png,
                "trajectory_plot_png": traj_png,
            },
        },
    }

    report_json_path = os.path.join(output_dir, "quality_assessment_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(quality_summary, f, indent=2)

    return quality_summary
