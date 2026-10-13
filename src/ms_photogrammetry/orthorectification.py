"""
Stages 5, 6, and 7: Digital Surface Elevation Interpolation, Collinearity Orthorectification,
Nadir-Prioritized Seamline Mosaicking, and Georeferenced Multiband GeoTIFF Export.
"""

from __future__ import annotations
import array
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .native import get_native_lib, array_ptr
from .metadata import CaptureEvent, utm_to_wgs84
from .radiometry import RadiometricCalibrationModel, calibrate_band_image
from .registration import MultispectralRegistrationModel
from .sfm import SfMResult
from .tiff_io import write_multiband_geotiff, write_png_rgb


@dataclass
class OrthomosaicProduct:
    width: int
    height: int
    origin_easting: float
    origin_northing: float
    pixel_size_m: float
    epsg_code: int
    valid_pixels: int
    total_pixels: int
    coverage_area_m2: float
    mean_overlap_observations: float
    dem_min_m: float
    dem_max_m: float
    dem_mean_m: float
    multiband_uint16_geotiff: str
    multiband_float32_geotiff: str
    ndvi_geotiff: str
    rgb_preview_png: str
    cir_preview_png: str
    ndvi_preview_png: str
    thermal_preview_png: str
    band_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


def build_dem_grid(
    sfm_result: SfMResult,
    default_ground_z: float,
    ortho_w: int,
    ortho_h: int,
    origin_e: float,
    origin_n: float,
    pixel_size_m: float,
    coarse_step: int = 32,
) -> Tuple[array.array, float, float, float]:
    """
    Build a smooth, physically continuous Digital Elevation Map (DEM) across the survey grid
    from Bundle-Adjusted multi-view 3D tie points:
      1. Filter multi-view tie points (>= 3 views, low reprojection error) using Median Absolute
         Deviation (MAD) to remove narrow-baseline ray triangulation outliers.
      2. Fit a global weighted linear terrain grade Z_plane(E, N) = c0 + c1*(E - E0) + c2*(N - N0)
         capturing the macro field drainage slope without edge step-cliffs.
      3. Interpolate smooth local terrain residuals using a wide Gaussian kernel (sigma = 14 m)
         on a coarse grid followed by bilinear interpolation to full resolution.
    """
    raw_pts = [
        (tp.easting, tp.northing, tp.elevation, len(tp.observations))
        for tp in sfm_result.tie_points
        if len(tp.observations) >= 3 and tp.mean_reproj_error_px < 5.5
    ]
    if len(raw_pts) < 15:
        raw_pts = [
            (tp.easting, tp.northing, tp.elevation, len(tp.observations))
            for tp in sfm_result.tie_points
            if len(tp.observations) >= 2
        ]

    if len(raw_pts) < 8:
        dem = array.array("f", [float(default_ground_z)]) * (ortho_w * ortho_h)
        return dem, default_ground_z, default_ground_z, default_ground_z

    # Robust MAD outlier rejection on tie-point elevations
    zs_sorted = sorted(p[2] for p in raw_pts)
    med_z = zs_sorted[len(zs_sorted) // 2]
    abs_devs = sorted(abs(z - med_z) for z in zs_sorted)
    mad_z = max(0.03, abs_devs[len(abs_devs) // 2] * 1.4826)
    z_clip = max(0.08, min(0.22, 2.2 * mad_z))

    good_pts = [
        (pe, pn, max(med_z - z_clip, min(med_z + z_clip, pz)), float(n_obs))
        for pe, pn, pz, n_obs in raw_pts
        if abs(pz - med_z) <= 3.0 * mad_z
    ]
    if len(good_pts) < 8:
        good_pts = [(pe, pn, med_z, 3.0) for pe, pn, _, _ in raw_pts]

    # Fit weighted planar drainage grade Z_trend(E, N) = med_z + b_e * (E - E_c) + b_n * (N - N_c)
    e_c = origin_e + 0.5 * ortho_w * pixel_size_m
    n_c = origin_n - 0.5 * ortho_h * pixel_size_m
    s_ee = 100.0  # ridge regularization to keep macro slope gentle (< 0.5%)
    s_nn = 100.0
    s_en = 0.0
    s_ez = 0.0
    s_nz = 0.0
    for pe, pn, pz, w_obs in good_pts:
        de = pe - e_c
        dn = pn - n_c
        dz = pz - med_z
        w = min(4.0, w_obs / 3.0)
        s_ee += w * de * de
        s_nn += w * dn * dn
        s_en += w * de * dn
        s_ez += w * de * dz
        s_nz += w * dn * dz

    det = s_ee * s_nn - s_en * s_en
    if abs(det) > 1e-6:
        b_e = (s_nn * s_ez - s_en * s_nz) / det
        b_n = (s_ee * s_nz - s_en * s_ez) / det
        # Clamp macro field slope to realistic graded turfgrass range (+/- 0.35% grade)
        b_e = max(-0.0035, min(0.0035, b_e))
        b_n = max(-0.0035, min(0.0035, b_n))
    else:
        b_e, b_n = 0.0, 0.0

    cw = (ortho_w + coarse_step - 1) // coarse_step + 1
    ch = (ortho_h + coarse_step - 1) // coarse_step + 1
    coarse_z = [[med_z for _ in range(cw)] for _ in range(ch)]

    z_min = 1e9
    z_max = -1e9
    z_sum = 0.0
    inv_2sigma2 = 1.0 / (2.0 * 14.0 * 14.0)  # 14m Gaussian spatial smoothing kernel

    for r in range(ch):
        n_coord = origin_n - (r * coarse_step) * pixel_size_m
        for c in range(cw):
            e_coord = origin_e + (c * coarse_step) * pixel_size_m
            z_trend = med_z + b_e * (e_coord - e_c) + b_n * (n_coord - n_c)
            w_sum = 0.50  # smooth anchor to planar trend at field perimeter (prevents edge cliffs)
            w_res_sum = 0.0
            for pe, pn, pz, w_obs in good_pts:
                d2 = (e_coord - pe) ** 2 + (n_coord - pn) ** 2
                if d2 < 900.0:  # 30m radius
                    w = (w_obs / 3.0) * math.exp(-d2 * inv_2sigma2)
                    p_trend = med_z + b_e * (pe - e_c) + b_n * (pn - n_c)
                    w_sum += w
                    w_res_sum += w * (pz - p_trend)
            # Blend macro planar drainage trend with 50% of smooth local tie-point residual
            z_val = z_trend + 0.50 * (w_res_sum / w_sum)
            coarse_z[r][c] = z_val
            z_min = min(z_min, z_val)
            z_max = max(z_max, z_val)
            z_sum += z_val

    z_mean = z_sum / (cw * ch)

    # Bilinearly interpolate coarse DEM to full grid resolution
    dem = array.array("f", b"\x00" * (ortho_w * ortho_h * 4))
    inv_step = 1.0 / coarse_step
    for row in range(ortho_h):
        ry = row * inv_step
        r0 = int(ry)
        r1 = min(ch - 1, r0 + 1)
        ay = ry - r0
        row_off = row * ortho_w
        c_row0 = coarse_z[r0]
        c_row1 = coarse_z[r1]
        for col in range(ortho_w):
            cx = col * inv_step
            c0 = int(cx)
            c1 = min(cw - 1, c0 + 1)
            ax = cx - c0
            v0 = c_row0[c0] * (1.0 - ax) + c_row0[c1] * ax
            v1 = c_row1[c0] * (1.0 - ax) + c_row1[c1] * ax
            dem[row_off + col] = v0 * (1.0 - ay) + v1 * ay

    return dem, z_min, z_max, z_mean


def generate_multispectral_orthomosaic(
    survey_captures: List[CaptureEvent],
    sfm_result: SfMResult,
    rad_model: RadiometricCalibrationModel,
    reg_model: MultispectralRegistrationModel,
    ground_elevation_m: float,
    output_dir: str,
    pixel_size_m: float = 0.015,
    bands_order: Tuple[int, ...] = (1, 2, 3, 5, 4, 7),
    external_dem_grid: Optional[array.array] = None,
) -> OrthomosaicProduct:
    """
    Execute Collinearity Orthorectification, Nadir-Prioritized Seamline Mosaicking,
    and Multiband GeoTIFF & Preview Generation using the Digital Elevation Map (DEM).

    Default spectral band order in the multiband GeoTIFF:
      1: Blue (475 nm)
      2: Green (560 nm)
      3: Red (668 nm)
      4: RedEdge (717 nm)
      5: NIR (842 nm)
      6: LWIR Thermal (11000 nm, degrees Celsius)
    """
    lib = get_native_lib()
    os.makedirs(output_dir, exist_ok=True)

    # Determine UTM Zone 14N bounding box covering all camera centers + half-footprint margin (10.5m)
    margin_m = 10.5
    eastings = [sfm_result.poses[c.capture_index].easting for c in survey_captures]
    northings = [sfm_result.poses[c.capture_index].northing for c in survey_captures]

    min_e = math.floor((min(eastings) - margin_m) * 10.0) / 10.0
    max_e = math.ceil((max(eastings) + margin_m) * 10.0) / 10.0
    min_n = math.floor((min(northings) - margin_m) * 10.0) / 10.0
    max_n = math.ceil((max(northings) + margin_m) * 10.0) / 10.0

    ortho_w = int(math.ceil((max_e - min_e) / pixel_size_m))
    ortho_h = int(math.ceil((max_n - min_n) / pixel_size_m))
    total_pixels = ortho_w * ortho_h

    origin_e = min_e
    origin_n = max_n  # Upper-left corner has minimum Easting and maximum Northing

    # Use pre-computed DEM grid if provided, otherwise build DEM surface grid first
    if external_dem_grid is not None and len(external_dem_grid) == total_pixels:
        dem_grid = external_dem_grid
        valid_dem = [float(dem_grid[i]) for i in range(0, total_pixels, 29) if dem_grid[i] > -900.0]
        dem_min = min(valid_dem) if valid_dem else ground_elevation_m
        dem_max = max(valid_dem) if valid_dem else ground_elevation_m
        dem_mean = (sum(valid_dem) / len(valid_dem)) if valid_dem else ground_elevation_m
    else:
        dem_grid, dem_min, dem_max, dem_mean = build_dem_grid(
            sfm_result,
            default_ground_z=ground_elevation_m,
            ortho_w=ortho_w,
            ortho_h=ortho_h,
            origin_e=origin_e,
            origin_n=origin_n,
            pixel_size_m=pixel_size_m,
        )

    # Allocate accumulation buffers for each band
    accum_weighted: Dict[int, array.array] = {
        b: array.array("f", b"\x00" * (total_pixels * 4)) for b in bands_order
    }
    accum_weight: Dict[int, array.array] = {
        b: array.array("f", b"\x00" * (total_pixels * 4)) for b in bands_order
    }
    min_r2 = array.array("f", [999.0]) * total_pixels
    obs_count = array.array("H", b"\x00" * (total_pixels * 2))

    # Process captures in parallel I/O batches of 12 to keep memory footprint low and CPU utilization high
    batch_size = 12
    for b_start in range(0, len(survey_captures), batch_size):
        batch = survey_captures[b_start : b_start + batch_size]

        def _calibrate_capture_bands(cap: CaptureEvent) -> Tuple[int, Dict[int, array.array]]:
            cal_dict: Dict[int, array.array] = {}
            for b in bands_order:
                if b in cap.bands:
                    cal_dict[b] = calibrate_band_image(cap.bands[b], rad_model=rad_model)
            return cap.capture_index, cal_dict

        with ThreadPoolExecutor(max_workers=batch_size) as ex:
            calibrated_batch = dict(ex.map(_calibrate_capture_bands, batch))

        for cap in batch:
            c_idx = cap.capture_index
            pose = sfm_result.poses[c_idx]
            rot_arr = array.array("d", pose.rotation_matrix_world_to_cam())
            cal_bands = calibrated_batch[c_idx]

            for b in bands_order:
                if b not in cal_bands:
                    continue
                mb = cap.bands[b]
                k1, k2, k3, p1, p2 = mb.distortion_k123_p12
                reg = reg_model.bands.get(b)
                bo_du = reg.boresight_du_px if reg else 0.0
                bo_dv = reg.boresight_dv_px if reg else 0.0

                # Track min_r2 and obs_count on the reference band (Band 2 Green)
                is_ref = (b == 2)
                lib.accumulate_ortho_frame(
                    array_ptr(cal_bands[b]),
                    mb.width,
                    mb.height,
                    mb.fx_px,
                    mb.fy_px,
                    mb.cx_px,
                    mb.cy_px,
                    k1,
                    k2,
                    k3,
                    p1,
                    p2,
                    array_ptr(rot_arr),
                    pose.easting,
                    pose.northing,
                    pose.altitude,
                    bo_du,
                    bo_dv,
                    array_ptr(dem_grid),
                    ortho_w,
                    ortho_h,
                    origin_e,
                    origin_n,
                    pixel_size_m,
                    pixel_size_m,
                    array_ptr(accum_weighted[b]),
                    array_ptr(accum_weight[b]),
                    array_ptr(min_r2) if is_ref else 0,
                    array_ptr(obs_count) if is_ref else 0,
                )

    # Finalize each band into Float32 and UInt16 buffers
    band_float32: Dict[int, array.array] = {}
    band_uint16: Dict[int, array.array] = {}
    band_descriptions: List[Dict[str, Any]] = []
    band_stats: Dict[str, Dict[str, float]] = {}
    valid_pixels = 0

    for b in bands_order:
        f32 = array.array("f", b"\x00" * (total_pixels * 4))
        u16 = array.array("H", b"\x00" * (total_pixels * 2))
        mb0 = survey_captures[0].bands[b]

        if b == 7:
            # LWIR thermal: float32 in Celsius; uint16 in centi-Kelvin (T_C + 273.15) * 100
            scale_u16 = 100.0
            offset_u16 = 273.15
            units_str = "celsius (float32) / centi-kelvin (uint16)"
        else:
            # Reflectance in [0..1]: uint16 scaled by 10000 (10000 = 100% reflectance, 0 = NoData)
            scale_u16 = 10000.0
            offset_u16 = 0.0
            units_str = "surface_reflectance (scale=10000 for uint16)"

        vp = lib.finalize_ortho_band(
            array_ptr(accum_weighted[b]),
            array_ptr(accum_weight[b]),
            total_pixels,
            scale_u16,
            offset_u16,
            -9999.0,
            array_ptr(f32),
            array_ptr(u16),
        )
        if b == 2:
            valid_pixels = vp

        band_float32[b] = f32
        band_uint16[b] = u16
        band_descriptions.append(
            {
                "band_index": b,
                "name": mb0.band_name,
                "central_wavelength_nm": mb0.central_wavelength_nm,
                "fwhm_nm": mb0.wavelength_fwhm_nm,
                "units": units_str,
            }
        )

        # Sample every 17th valid pixel for fast summary statistics
        samples_b = [f32[i] for i in range(0, total_pixels, 17) if f32[i] > -900.0]
        if samples_b:
            samples_b.sort()
            mean_b = sum(samples_b) / len(samples_b)
            var_b = sum((x - mean_b) ** 2 for x in samples_b) / len(samples_b)
            band_stats[mb0.band_name] = {
                "min": round(samples_b[0], 4),
                "p05": round(samples_b[int(len(samples_b) * 0.05)], 4),
                "median": round(samples_b[len(samples_b) // 2], 4),
                "mean": round(mean_b, 4),
                "p95": round(samples_b[int(len(samples_b) * 0.95)], 4),
                "max": round(samples_b[-1], 4),
                "std": round(math.sqrt(var_b), 4),
            }

    # Compute NDVI, NDRE, and RGBA visualization buffers in C++
    ndvi_f32 = array.array("f", b"\x00" * (total_pixels * 4))
    ndre_f32 = array.array("f", b"\x00" * (total_pixels * 4))
    rgb_rgba = array.array("B", b"\x00" * (total_pixels * 4))
    cir_rgba = array.array("B", b"\x00" * (total_pixels * 4))
    ndvi_rgba = array.array("B", b"\x00" * (total_pixels * 4))
    thermal_rgba = array.array("B", b"\x00" * (total_pixels * 4))

    lib.compute_ndvi_and_rgb_previews(
        array_ptr(band_float32[1]),
        array_ptr(band_float32[2]),
        array_ptr(band_float32[3]),
        array_ptr(band_float32[4]),
        array_ptr(band_float32[5]),
        array_ptr(band_float32[7]) if 7 in band_float32 else 0,
        array_ptr(accum_weight[2]),
        ortho_w,
        ortho_h,
        array_ptr(ndvi_f32),
        array_ptr(ndre_f32),
        array_ptr(rgb_rgba),
        array_ptr(cir_rgba),
        array_ptr(ndvi_rgba),
        array_ptr(thermal_rgba),
    )

    ndvi_samples = [ndvi_f32[i] for i in range(0, total_pixels, 17) if ndvi_f32[i] > -900.0]
    if ndvi_samples:
        ndvi_samples.sort()
        mean_ndvi = sum(ndvi_samples) / len(ndvi_samples)
        band_stats["NDVI"] = {
            "min": round(ndvi_samples[0], 4),
            "p05": round(ndvi_samples[int(len(ndvi_samples) * 0.05)], 4),
            "median": round(ndvi_samples[len(ndvi_samples) // 2], 4),
            "mean": round(mean_ndvi, 4),
            "p95": round(ndvi_samples[int(len(ndvi_samples) * 0.95)], 4),
            "max": round(ndvi_samples[-1], 4),
        }

    # Write Multiband UInt16 GeoTIFF (primary compact deliverable, 0 = NoData)
    uint16_geotiff_path = os.path.join(output_dir, "multispectral_orthomosaic_uint16.tif")
    write_multiband_geotiff(
        filepath=uint16_geotiff_path,
        bands_data=[band_uint16[b].tobytes() for b in bands_order],
        width=ortho_w,
        height=ortho_h,
        origin_easting=origin_e,
        origin_northing=origin_n,
        pixel_size_x=pixel_size_m,
        pixel_size_y=pixel_size_m,
        band_descriptions=band_descriptions,
        sample_format="uint16",
        epsg_code=32614,
        crs_citation="WGS 84 / UTM zone 14N",
        nodata_value="0",
    )

    # Write Multiband Float32 Reflectance GeoTIFF (scientific floating-point deliverable, -9999 = NoData)
    float32_geotiff_path = os.path.join(output_dir, "multispectral_orthomosaic_float32.tif")
    write_multiband_geotiff(
        filepath=float32_geotiff_path,
        bands_data=[band_float32[b].tobytes() for b in bands_order],
        width=ortho_w,
        height=ortho_h,
        origin_easting=origin_e,
        origin_northing=origin_n,
        pixel_size_x=pixel_size_m,
        pixel_size_y=pixel_size_m,
        band_descriptions=band_descriptions,
        sample_format="float32",
        epsg_code=32614,
        crs_citation="WGS 84 / UTM zone 14N",
        nodata_value="-9999",
    )

    # Write 2-band Float32 Vegetation Index GeoTIFF (Band 1: NDVI, Band 2: NDRE)
    ndvi_geotiff_path = os.path.join(output_dir, "multispectral_indices_ndvi_ndre.tif")
    write_multiband_geotiff(
        filepath=ndvi_geotiff_path,
        bands_data=[ndvi_f32.tobytes(), ndre_f32.tobytes()],
        width=ortho_w,
        height=ortho_h,
        origin_easting=origin_e,
        origin_northing=origin_n,
        pixel_size_x=pixel_size_m,
        pixel_size_y=pixel_size_m,
        band_descriptions=[
            {"name": "NDVI", "central_wavelength_nm": "", "fwhm_nm": "", "units": "dimensionless [-1, 1]"},
            {"name": "NDRE", "central_wavelength_nm": "", "fwhm_nm": "", "units": "dimensionless [-1, 1]"},
        ],
        sample_format="float32",
        epsg_code=32614,
        crs_citation="WGS 84 / UTM zone 14N",
        nodata_value="-9999",
    )

    # Export RGBA PNG visual composites
    rgb_png_path = os.path.join(output_dir, "orthomosaic_rgb_preview.png")
    cir_png_path = os.path.join(output_dir, "orthomosaic_cir_false_color.png")
    ndvi_png_path = os.path.join(output_dir, "orthomosaic_ndvi_map.png")
    thermal_png_path = os.path.join(output_dir, "orthomosaic_lwir_thermal.png")

    write_png_rgb(rgb_png_path, rgb_rgba.tobytes(), ortho_w, ortho_h)
    write_png_rgb(cir_png_path, cir_rgba.tobytes(), ortho_w, ortho_h)
    write_png_rgb(ndvi_png_path, ndvi_rgba.tobytes(), ortho_w, ortho_h)
    write_png_rgb(thermal_png_path, thermal_rgba.tobytes(), ortho_w, ortho_h)

    obs_samples = [obs_count[i] for i in range(0, total_pixels, 13) if obs_count[i] > 0]
    mean_obs = sum(obs_samples) / max(1, len(obs_samples))
    coverage_area = valid_pixels * (pixel_size_m * pixel_size_m)

    return OrthomosaicProduct(
        width=ortho_w,
        height=ortho_h,
        origin_easting=origin_e,
        origin_northing=origin_n,
        pixel_size_m=pixel_size_m,
        epsg_code=32614,
        valid_pixels=valid_pixels,
        total_pixels=total_pixels,
        coverage_area_m2=coverage_area,
        mean_overlap_observations=mean_obs,
        dem_min_m=dem_min,
        dem_max_m=dem_max,
        dem_mean_m=dem_mean,
        multiband_uint16_geotiff=uint16_geotiff_path,
        multiband_float32_geotiff=float32_geotiff_path,
        ndvi_geotiff=ndvi_geotiff_path,
        rgb_preview_png=rgb_png_path,
        cir_preview_png=cir_png_path,
        ndvi_preview_png=ndvi_png_path,
        thermal_preview_png=thermal_png_path,
        band_stats=band_stats,
    )
