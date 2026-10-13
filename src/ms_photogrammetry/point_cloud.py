"""
Stage 9: Dense Multi-View Stereo (MVS) 3D Point Cloud Reconstruction & 3D Canopy Metrics.
Implemented from scratch using pure Python standard library (struct, array, math, csv, json)
and the custom C++17 native photogrammetry engine (reconstruct_mvs_dense_dsm,
compute_dtm_chm_and_3d_renders).

Generates:
  1. Dense Photogrammetric 3D Point Cloud in ASPRS LAS 1.2 binary format
     (`output/dense_photogrammetric_point_cloud.las` and root `photogrammetric_point_cloud.las`)
  2. Multi-Attribute Scientific 3D Point Cloud in Stanford PLY binary format
     (`output/dense_photogrammetric_point_cloud.ply` with X, Y, Z, Normals, RGB, CHM, NDVI, NDRE, Thermal, ZNCC)
  3. Sparse SfM 3D Tie-Point Cloud (`output/sparse_sfm_tiepoints.ply` & `.csv`)
  4. 5-Band Georeferenced 3D Metric Raster GeoTIFF (`output/photogrammetric_3d_dsm_dtm_chm.tif`:
     Band 1: DSM, Band 2: Bare-Earth DTM, Band 3: Canopy Height Model CHM, Band 4: Slope, Band 5: 3D Rugosity)
  5. Plot-Level 3D Structural & Spectral Metrics Table (`output/plot_level_3d_metrics.csv`)
  6. Comprehensive 3D Metrics Summary JSON (`output/point_cloud_3d_metrics_report.json`)
  7. Visual 3D Products (`point_cloud_dsm_hillshade.png`, `point_cloud_chm_map.png`, `point_cloud_3d_perspective_render.png`)
"""

from __future__ import annotations
import array
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import datetime
import json
import math
import os
import struct
from typing import Dict, List, Tuple, Any

from .metadata import CaptureEvent
from .native import get_native_lib, array_ptr
from .orthorectification import build_dem_grid
from .radiometry import RadiometricCalibrationModel, calibrate_band_image
from .sfm import SfMResult
from .tiff_io import write_multiband_geotiff, write_png_rgb


@dataclass
class PointCloud3DResult:
    las_filepath: str
    ply_filepath: str
    sparse_ply_filepath: str
    dsm_dtm_chm_geotiff: str
    plot_metrics_csv: str
    metrics_json_filepath: str
    dsm_hillshade_png: str
    chm_map_png: str
    perspective_3d_png: str
    num_dense_points: int
    num_sparse_points: int
    summary_metrics: Dict[str, Any]


def _read_multiband_float32_geotiff_bands(filepath: str, width: int, height: int, num_bands: int) -> List[array.array]:
    """Read planar float32 bands from our generated multiband GeoTIFF using pure Python struct."""
    npix = width * height
    with open(filepath, "rb") as f:
        sig, ver, ifd_off = struct.unpack("<2sHI", f.read(8))
        f.seek(ifd_off)
        num_entries = struct.unpack("<H", f.read(2))[0]
        strip_offsets = 0
        for _ in range(num_entries):
            tag, ttype, count, val_off = struct.unpack("<HHII", f.read(12))
            if tag == 273:
                pos = f.tell()
                f.seek(val_off)
                strip_offsets = struct.unpack("<I", f.read(4))[0]
                f.seek(pos)
        f.seek(strip_offsets)
        raw = f.read(npix * num_bands * 4)

    bands: List[array.array] = []
    for b in range(num_bands):
        arr = array.array("f", raw[b * npix * 4 : (b + 1) * npix * 4])
        bands.append(arr)
    return bands


def write_sparse_tiepoints_ply_and_csv(
    sfm_result: SfMResult,
    ply_filepath: str,
    csv_filepath: str,
) -> int:
    """Export the Bundle-Adjusted 3D SfM tie points to binary PLY and CSV."""
    tps = sfm_result.tie_points
    n_pts = len(tps)

    # Write CSV
    with open(csv_filepath, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["point_id", "easting_m", "northing_m", "elevation_m", "num_observations", "reproj_error_px"])
        for idx, tp in enumerate(tps):
            writer.writerow([
                idx,
                f"{tp.easting:.4f}",
                f"{tp.northing:.4f}",
                f"{tp.elevation:.4f}",
                len(tp.observations),
                f"{tp.mean_reproj_error_px:.3f}",
            ])

    # Write binary little-endian PLY
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Generated from scratch by ms_photogrammetry SfM Bundle Adjustment\n"
        "comment CRS: EPSG:32614 (WGS 84 / UTM zone 14N)\n"
        f"element vertex {n_pts}\n"
        "property double x\n"
        "property double y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property ushort num_observations\n"
        "property float reproj_error_px\n"
        "end_header\n"
    ).encode("ascii")

    rec_struct = struct.Struct("<ddfBBBHf")
    buf = bytearray(n_pts * rec_struct.size)
    offset = 0
    for tp in tps:
        # Color tie points by elevation (hypsometric tint)
        t = max(0.0, min(1.0, (tp.elevation - 211.4) / 0.9))
        r = int(50 + 190 * t)
        g = int(180 - 60 * abs(2.0 * t - 1.0))
        b = int(230 - 180 * t)
        rec_struct.pack_into(
            buf,
            offset,
            tp.easting,
            tp.northing,
            float(tp.elevation),
            r,
            g,
            b,
            len(tp.observations),
            float(tp.mean_reproj_error_px),
        )
        offset += rec_struct.size

    with open(ply_filepath, "wb") as pf:
        pf.write(header)
        pf.write(buf)

    return n_pts


def write_las12_point_cloud(
    filepath: str,
    grid_w: int,
    grid_h: int,
    origin_e: float,
    origin_n: float,
    cell_size_m: float,
    dsm_grid: array.array,
    nir_grid: array.array,
    rgb_rgba: array.array,
    class_grid: array.array,
    num_views_grid: array.array,
) -> int:
    """
    Write an industry-standard ASPRS LAS 1.2 binary point cloud (Point Data Record Format 2,
    26 bytes per point with X, Y, Z, Intensity, Return Info, Classification, and 16-bit RGB)
    from scratch using Python's standard library `struct` module.
    """
    # Count valid points and compute bounding box
    valid_indices = [i for i in range(grid_w * grid_h) if dsm_grid[i] > -900.0]
    n_pts = len(valid_indices)
    if n_pts == 0:
        return 0

    min_x = origin_e
    max_x = origin_e + grid_w * cell_size_m
    min_y = origin_n - grid_h * cell_size_m
    max_y = origin_n
    min_z = min(float(dsm_grid[i]) for i in valid_indices)
    max_z = max(float(dsm_grid[i]) for i in valid_indices)

    scale_x = 0.0001  # 0.1 mm precision
    scale_y = 0.0001
    scale_z = 0.0001
    off_x = math.floor(min_x)
    off_y = math.floor(min_y)
    off_z = math.floor(min_z)

    now = datetime.datetime.now()
    day_of_year = now.timetuple().tm_yday
    year = now.year

    # ASPRS LAS 1.2 Public Header Block is exactly 227 bytes
    sys_id = b"MS_PHOTOGRAMMETRY_CPP17".ljust(32, b"\x00")[:32]
    gen_sw = b"FROM_SCRATCH_MVS_PIPELINE_1.0".ljust(32, b"\x00")[:32]

    header_fmt = "<4sHHIHH8sBB32s32sHHHIIBHI5I3d3d6d"
    header_bytes = struct.pack(
        header_fmt,
        b"LASF",               # File Signature ("LASF")
        0,                     # File Source ID
        0,                     # Global Encoding
        0, 0, 0, b"\x00" * 8,  # Project ID GUID
        1,                     # Version Major (1)
        2,                     # Version Minor (2)
        sys_id,                # System Identifier (32 bytes)
        gen_sw,                # Generating Software (32 bytes)
        day_of_year,           # File Creation Day of Year
        year,                  # File Creation Year
        227,                   # Header Size (227 bytes)
        227,                   # Offset to point data (no VLRs -> 227)
        0,                     # Number of Variable Length Records
        2,                     # Point Data Format ID (Format 2: XYZ + Intensity + Class + RGB)
        26,                    # Point Data Record Length (26 bytes)
        n_pts,                 # Number of point records
        n_pts, 0, 0, 0, 0,     # Number of points by return (1..5)
        scale_x, scale_y, scale_z,
        off_x, off_y, off_z,
        max_x, min_x, max_y, min_y, max_z, min_z,
    )
    assert len(header_bytes) == 227, f"LAS 1.2 header must be 227 bytes, got {len(header_bytes)}"

    # Pack Point Data Record Format 2 (26 bytes per point):
    # X (i4), Y (i4), Z (i4), Intensity (u2), ReturnByte (u1), Classification (u1),
    # ScanAngleRank (i1), UserData (u1), PointSourceID (u2), Red (u2), Green (u2), Blue (u2)
    pt_struct = struct.Struct("<iiiHBBbBHHHH")
    buf = bytearray(n_pts * 26)
    offset = 0

    inv_scale = 10000.0
    for idx in valid_indices:
        r = idx // grid_w
        c = idx % grid_w
        east = origin_e + (c + 0.5) * cell_size_m
        north = origin_n - (r + 0.5) * cell_size_m
        z = float(dsm_grid[idx])

        xi = int(round((east - off_x) * inv_scale))
        yi = int(round((north - off_y) * inv_scale))
        zi = int(round((z - off_z) * inv_scale))

        # Store calibrated NIR reflectance (scaled 0..65535) in LAS Intensity
        nir_val = max(0.0, min(1.0, float(nir_grid[idx])))
        intensity = int(round(nir_val * 65535.0))

        cls_code = int(class_grid[idx])
        n_views = min(255, int(num_views_grid[idx]))

        # 16-bit RGB as required by LAS 1.2 specification (8-bit channel * 257)
        red16 = int(rgb_rgba[4 * idx + 0]) * 257
        green16 = int(rgb_rgba[4 * idx + 1]) * 257
        blue16 = int(rgb_rgba[4 * idx + 2]) * 257

        pt_struct.pack_into(
            buf,
            offset,
            xi,
            yi,
            zi,
            intensity,
            0b00001001,  # Return 1 of 1
            cls_code,
            0,           # Scan Angle Rank
            n_views,     # User Data = Number of overlapping stereo views
            1,           # Point Source ID
            red16,
            green16,
            blue16,
        )
        offset += 26

    with open(filepath, "wb") as f:
        f.write(header_bytes)
        f.write(buf)

    return n_pts


def write_ply_point_cloud(
    filepath: str,
    grid_w: int,
    grid_h: int,
    origin_e: float,
    origin_n: float,
    cell_size_m: float,
    dsm_grid: array.array,
    chm_grid: array.array,
    normals_grid: array.array,
    rgb_rgba: array.array,
    ndvi_grid: array.array,
    ndre_grid: array.array,
    thermal_grid: array.array,
    zncc_grid: array.array,
    class_grid: array.array,
) -> int:
    """
    Write a rich binary little-endian Stanford PLY 3D point cloud containing:
      - Full UTM Zone 14N coordinates (double x, double y, float z)
      - 3D surface normals (float nx, ny, nz)
      - True-color sRGB (uchar red, green, blue)
      - 3D & Spectral attributes (float canopy_height_m, float ndvi, float ndre,
        float temperature_c, float zncc_confidence, uchar classification)
    """
    valid_indices = [i for i in range(grid_w * grid_h) if dsm_grid[i] > -900.0]
    n_pts = len(valid_indices)
    if n_pts == 0:
        return 0

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Generated from scratch by ms_photogrammetry Dense MVS Pipeline\n"
        "comment CRS: EPSG:32614 (WGS 84 / UTM zone 14N)\n"
        f"element vertex {n_pts}\n"
        "property double x\n"
        "property double y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float canopy_height_m\n"
        "property float ndvi\n"
        "property float ndre\n"
        "property float temperature_c\n"
        "property float zncc_confidence\n"
        "property uchar classification\n"
        "end_header\n"
    ).encode("ascii")

    rec_struct = struct.Struct("<ddffffBBBfffffB")
    buf = bytearray(n_pts * rec_struct.size)
    offset = 0

    for idx in valid_indices:
        r = idx // grid_w
        c = idx % grid_w
        east = origin_e + (c + 0.5) * cell_size_m
        north = origin_n - (r + 0.5) * cell_size_m

        rec_struct.pack_into(
            buf,
            offset,
            east,
            north,
            float(dsm_grid[idx]),
            float(normals_grid[3 * idx + 0]),
            float(normals_grid[3 * idx + 1]),
            float(normals_grid[3 * idx + 2]),
            int(rgb_rgba[4 * idx + 0]),
            int(rgb_rgba[4 * idx + 1]),
            int(rgb_rgba[4 * idx + 2]),
            float(chm_grid[idx]),
            float(ndvi_grid[idx]),
            float(ndre_grid[idx]),
            float(thermal_grid[idx]),
            float(zncc_grid[idx]),
            int(class_grid[idx]),
        )
        offset += rec_struct.size

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(buf)

    return n_pts


def generate_photogrammetric_point_cloud_and_3d_metrics(
    survey_captures: List[CaptureEvent],
    sfm_result: SfMResult,
    rad_model: RadiometricCalibrationModel,
    ground_elevation_m: float,
    output_dir: str,
    ortho_pixel_size_m: float = 0.015,
    point_spacing_m: float = 0.06,
) -> PointCloud3DResult:
    """
    Execute Dense Multi-View Stereo (MVS) 3D Point Cloud Reconstruction, Bare-Earth DTM / CHM
    separation, 3D Structural & Spectral Plot Metrics computation, and ASPRS LAS 1.2 + PLY export.
    """
    lib = get_native_lib()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export Sparse SfM 3D Tie-Point Cloud first
    sparse_ply_path = os.path.join(output_dir, "sparse_sfm_tiepoints.ply")
    sparse_csv_path = os.path.join(output_dir, "sparse_sfm_tiepoints.csv")
    num_sparse = write_sparse_tiepoints_ply_and_csv(sfm_result, sparse_ply_path, sparse_csv_path)

    # 2. Determine grid geometry matching the orthomosaic extent
    margin_m = 10.5
    eastings = [sfm_result.poses[c.capture_index].easting for c in survey_captures]
    northings = [sfm_result.poses[c.capture_index].northing for c in survey_captures]

    min_e = math.floor((min(eastings) - margin_m) * 10.0) / 10.0
    max_e = math.ceil((max(eastings) + margin_m) * 10.0) / 10.0
    min_n = math.floor((min(northings) - margin_m) * 10.0) / 10.0
    max_n = math.ceil((max(northings) + margin_m) * 10.0) / 10.0

    ortho_w = int(math.ceil((max_e - min_e) / ortho_pixel_size_m))
    ortho_h = int(math.ceil((max_n - min_n) / ortho_pixel_size_m))
    origin_e = min_e
    origin_n = max_n

    scale_step = max(1, int(round(point_spacing_m / ortho_pixel_size_m)))
    cell_size_m = ortho_pixel_size_m * scale_step
    grid_w = ortho_w // scale_step
    grid_h = ortho_h // scale_step
    total_cells = grid_w * grid_h

    # 3. Sample co-registered multispectral + thermal attributes from the calibrated float32 orthomosaic
    f32_ortho_path = os.path.join(output_dir, "multispectral_orthomosaic_float32.tif")
    ortho_bands = _read_multiband_float32_geotiff_bands(f32_ortho_path, ortho_w, ortho_h, 6)
    b_full, g_full, r_full, re_full, nir_full, tc_full = ortho_bands

    grid_b = array.array("f", [0.0]) * total_cells
    grid_g = array.array("f", [0.0]) * total_cells
    grid_r = array.array("f", [0.0]) * total_cells
    grid_re = array.array("f", [0.0]) * total_cells
    grid_nir = array.array("f", [0.0]) * total_cells
    grid_tc = array.array("f", [0.0]) * total_cells
    grid_ndvi = array.array("f", [-9999.0]) * total_cells
    grid_ndre = array.array("f", [-9999.0]) * total_cells
    grid_rgb_rgba = array.array("B", [0]) * (total_cells * 4)
    valid_mask = array.array("B", [0]) * total_cells

    for r in range(grid_h):
        orow = min(ortho_h - 1, r * scale_step + scale_step // 2)
        orow_off = orow * ortho_w
        grow_off = r * grid_w
        for c in range(grid_w):
            ocol = min(ortho_w - 1, c * scale_step + scale_step // 2)
            oidx = orow_off + ocol
            gidx = grow_off + c

            bv = b_full[oidx]
            gv = g_full[oidx]
            rv = r_full[oidx]
            rev = re_full[oidx]
            nv = nir_full[oidx]
            tv = tc_full[oidx]

            if bv > 0.0 and gv > 0.0 and rv > 0.0 and rev > 0.0 and nv > 0.0:
                valid_mask[gidx] = 1
                grid_b[gidx] = bv
                grid_g[gidx] = gv
                grid_r[gidx] = rv
                grid_re[gidx] = rev
                grid_nir[gidx] = nv
                grid_tc[gidx] = tv
                ndvi = (nv - rv) / (nv + rv) if (nv + rv) > 1e-5 else 0.0
                ndre = (nv - rev) / (nv + rev) if (nv + rev) > 1e-5 else 0.0
                grid_ndvi[gidx] = max(-1.0, min(1.0, ndvi))
                grid_ndre[gidx] = max(-1.0, min(1.0, ndre))

                # Gamma-stretched sRGB for point cloud coloring
                sr = int(min(255.0, max(0.0, ((max(0.0, rv - 0.01) / 0.15) ** 0.58) * 255.0)))
                sg = int(min(255.0, max(0.0, ((max(0.0, gv - 0.015) / 0.145) ** 0.58) * 255.0)))
                sb = int(min(255.0, max(0.0, ((max(0.0, bv - 0.01) / 0.11) ** 0.58) * 255.0)))
                grid_rgb_rgba[4 * gidx + 0] = sr
                grid_rgb_rgba[4 * gidx + 1] = sg
                grid_rgb_rgba[4 * gidx + 2] = sb
                grid_rgb_rgba[4 * gidx + 3] = 255

    # 4. Build initial prior surface from triangulated 3D tie points on (grid_w, grid_h)
    prior_dem, _, _, _ = build_dem_grid(
        sfm_result,
        default_ground_z=ground_elevation_m,
        ortho_w=grid_w,
        ortho_h=grid_h,
        origin_e=origin_e,
        origin_n=origin_n,
        pixel_size_m=cell_size_m,
    )

    # 5. Pre-undistort and downsample Reference Band 2 (Green) for all 216 survey captures
    # Working MVS image resolution: 516 x 386 (downsample factor 4 -> 3.34 cm/px)
    ds_factor = 4
    mb0 = survey_captures[0].bands[2]
    mvs_w = mb0.width // ds_factor
    mvs_h = mb0.height // ds_factor
    mvs_fx = mb0.fx_px / ds_factor
    mvs_fy = mb0.fy_px / ds_factor
    mvs_cx = mb0.cx_px / ds_factor
    mvs_cy = mb0.cy_px / ds_factor
    k1, k2, k3, p1, p2 = mb0.distortion_k123_p12

    num_cams = len(survey_captures)
    img_stride = mvs_w * mvs_h
    all_undist_images = array.array("f", b"\x00" * (num_cams * img_stride * 4))
    cam_poses_arr = array.array("d", [0.0] * (num_cams * 12))

    def _prep_mvs_cam(idx_cap: Tuple[int, CaptureEvent]) -> Tuple[int, array.array]:
        k_idx, cap = idx_cap
        mb = cap.bands[2]
        cal_full = calibrate_band_image(mb, rad_model=rad_model)
        # Box-downsample by 4 into (mvs_w, mvs_h) and undistort in C++
        ds_raw = array.array("f", b"\x00" * (img_stride * 4))
        w_full = mb.width
        for y in range(mvs_h):
            r0 = (y * 4 + 1) * w_full
            r1 = (y * 4 + 2) * w_full
            out_row = y * mvs_w
            for x in range(mvs_w):
                c0 = x * 4 + 1
                ds_raw[out_row + x] = 0.25 * (
                    cal_full[r0 + c0] + cal_full[r0 + c0 + 1] + cal_full[r1 + c0] + cal_full[r1 + c0 + 1]
                )
        undist = array.array("f", b"\x00" * (img_stride * 4))
        lib.undistort_image_float(
            array_ptr(ds_raw),
            mvs_w,
            mvs_h,
            mvs_fx,
            mvs_fy,
            mvs_cx,
            mvs_cy,
            k1,
            k2,
            k3,
            p1,
            p2,
            array_ptr(undist),
        )
        return k_idx, undist

    with ThreadPoolExecutor(max_workers=8) as ex:
        for k_idx, undist in ex.map(_prep_mvs_cam, enumerate(survey_captures)):
            start = k_idx * img_stride
            all_undist_images[start : start + img_stride] = undist
            pose = sfm_result.poses[survey_captures[k_idx].capture_index]
            rot = pose.rotation_matrix_world_to_cam()
            base = k_idx * 12
            cam_poses_arr[base + 0] = pose.easting
            cam_poses_arr[base + 1] = pose.northing
            cam_poses_arr[base + 2] = pose.altitude
            for r_i in range(9):
                cam_poses_arr[base + 3 + r_i] = rot[r_i]

    # 6. Run C++17 Multi-View Stereo (MVS) Plane-Sweep ZNCC Dense Reconstruction around the DEM
    dsm_grid = array.array("f", [-9999.0]) * total_cells
    zncc_grid = array.array("f", [0.0]) * total_cells
    num_views_grid = array.array("H", [0]) * total_cells

    lib.reconstruct_mvs_dense_dsm(
        array_ptr(all_undist_images),
        num_cams,
        mvs_w,
        mvs_h,
        mvs_fx,
        mvs_fy,
        mvs_cx,
        mvs_cy,
        array_ptr(cam_poses_arr),
        array_ptr(prior_dem),
        array_ptr(valid_mask),
        grid_w,
        grid_h,
        origin_e,
        origin_n,
        cell_size_m,
        0.14,  # +/- 14 cm plane-sweep search window around the Digital Elevation Map
        17,    # 17 elevation planes + parabolic sub-step refinement
        array_ptr(dsm_grid),
        array_ptr(zncc_grid),
        array_ptr(num_views_grid),
    )

    # 7. Compute Bare-Earth Digital Elevation Map (DTM) FIRST, then CHM, normalize 3D Point Cloud
    #    elevations (dsm_grid = dtm_grid + chm_grid in-place), and render 3D products
    dtm_grid = array.array("f", [-9999.0]) * total_cells
    chm_grid = array.array("f", [-9999.0]) * total_cells
    slope_grid = array.array("f", [-9999.0]) * total_cells
    rugosity_grid = array.array("f", [-9999.0]) * total_cells
    normals_grid = array.array("f", [0.0]) * (total_cells * 3)
    class_grid = array.array("B", [0]) * total_cells
    dsm_hillshade_rgba = array.array("B", [0]) * (total_cells * 4)
    chm_rgba = array.array("B", [0]) * (total_cells * 4)
    persp_w, persp_h = 1600, 1000
    persp_rgba = array.array("B", [0]) * (persp_w * persp_h * 4)

    lib.compute_dtm_chm_and_3d_renders(
        array_ptr(dsm_grid),
        array_ptr(grid_ndvi),
        array_ptr(grid_rgb_rgba),
        grid_w,
        grid_h,
        cell_size_m,
        array_ptr(dtm_grid),
        array_ptr(chm_grid),
        array_ptr(slope_grid),
        array_ptr(rugosity_grid),
        array_ptr(normals_grid),
        array_ptr(class_grid),
        array_ptr(dsm_hillshade_rgba),
        array_ptr(chm_rgba),
        persp_w,
        persp_h,
        array_ptr(persp_rgba),
    )

    # 8. Export ASPRS LAS 1.2 and Stanford PLY Point Clouds (using DEM-normalized dsm_grid)
    las_path = os.path.join(output_dir, "dense_photogrammetric_point_cloud.las")
    num_dense_pts = write_las12_point_cloud(
        las_path,
        grid_w,
        grid_h,
        origin_e,
        origin_n,
        cell_size_m,
        dsm_grid,
        grid_nir,
        grid_rgb_rgba,
        class_grid,
        num_views_grid,
    )

    # Also copy primary LAS point cloud to workspace root for immediate visibility
    root_las_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "photogrammetric_point_cloud.las")
    try:
        import shutil
        shutil.copyfile(las_path, root_las_path)
    except Exception:
        pass

    ply_path = os.path.join(output_dir, "dense_photogrammetric_point_cloud.ply")
    write_ply_point_cloud(
        ply_path,
        grid_w,
        grid_h,
        origin_e,
        origin_n,
        cell_size_m,
        dsm_grid,
        chm_grid,
        normals_grid,
        grid_rgb_rgba,
        grid_ndvi,
        grid_ndre,
        grid_tc,
        zncc_grid,
        class_grid,
    )

    # 9. Export 5-Band Georeferenced 3D Metric GeoTIFF (DSM, DTM, CHM, Slope, Rugosity)
    #    and standalone Digital Elevation Map GeoTIFF (digital_elevation_map.tif)
    dsm_geotiff_path = os.path.join(output_dir, "photogrammetric_3d_dsm_dtm_chm.tif")
    write_multiband_geotiff(
        filepath=dsm_geotiff_path,
        bands_data=[
            dsm_grid.tobytes(),
            dtm_grid.tobytes(),
            chm_grid.tobytes(),
            slope_grid.tobytes(),
            rugosity_grid.tobytes(),
        ],
        width=grid_w,
        height=grid_h,
        origin_easting=origin_e,
        origin_northing=origin_n,
        pixel_size_x=cell_size_m,
        pixel_size_y=cell_size_m,
        band_descriptions=[
            {"band_index": 1, "name": "DSM_Elevation_m", "units": "meters_MSL"},
            {"band_index": 2, "name": "DTM_Bare_Earth_m", "units": "meters_MSL"},
            {"band_index": 3, "name": "CHM_Canopy_Height_m", "units": "meters_AGL"},
            {"band_index": 4, "name": "Surface_Slope_deg", "units": "degrees"},
            {"band_index": 5, "name": "Surface_Rugosity_3D_2D_Ratio", "units": "dimensionless"},
        ],
        sample_format="float32",
        epsg_code=32614,
        crs_citation="WGS 84 / UTM zone 14N",
        nodata_value="-9999",
    )

    dem_geotiff_path = os.path.join(output_dir, "digital_elevation_map_dem.tif")
    write_multiband_geotiff(
        filepath=dem_geotiff_path,
        bands_data=[
            dtm_grid.tobytes(),
            dsm_grid.tobytes(),
        ],
        width=grid_w,
        height=grid_h,
        origin_easting=origin_e,
        origin_northing=origin_n,
        pixel_size_x=cell_size_m,
        pixel_size_y=cell_size_m,
        band_descriptions=[
            {"band_index": 1, "name": "DEM_Bare_Earth_DTM_m", "units": "meters_MSL"},
            {"band_index": 2, "name": "DSM_Surface_Elevation_m", "units": "meters_MSL"},
        ],
        sample_format="float32",
        epsg_code=32614,
        crs_citation="WGS 84 / UTM zone 14N",
        nodata_value="-9999",
    )
    root_dem_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "digital_elevation_map.tif")
    try:
        import shutil
        shutil.copyfile(dem_geotiff_path, root_dem_path)
    except Exception:
        pass

    # 10. Write 3D Visual PNGs (Hillshaded DSM, CHM Map, 3D Oblique Perspective Render)
    dsm_hillshade_png = os.path.join(output_dir, "point_cloud_dsm_hillshade.png")
    chm_map_png = os.path.join(output_dir, "point_cloud_chm_map.png")
    persp_3d_png = os.path.join(output_dir, "point_cloud_3d_perspective_render.png")

    write_png_rgb(dsm_hillshade_png, dsm_hillshade_rgba.tobytes(), grid_w, grid_h)
    write_png_rgb(chm_map_png, chm_rgba.tobytes(), grid_w, grid_h)
    write_png_rgb(persp_3d_png, persp_rgba.tobytes(), persp_w, persp_h)

    # 11. Compute Plot-Level & Field-Wide 3D Structural & Spectral Metrics
    plot_csv_path = os.path.join(output_dir, "plot_level_3d_metrics.csv")
    summary_metrics = _compute_3d_and_plot_metrics(
        grid_w=grid_w,
        grid_h=grid_h,
        origin_e=origin_e,
        origin_n=origin_n,
        cell_size_m=cell_size_m,
        dsm_grid=dsm_grid,
        dtm_grid=dtm_grid,
        chm_grid=chm_grid,
        slope_grid=slope_grid,
        rugosity_grid=rugosity_grid,
        ndvi_grid=grid_ndvi,
        ndre_grid=grid_ndre,
        tc_grid=grid_tc,
        zncc_grid=zncc_grid,
        class_grid=class_grid,
        num_views_grid=num_views_grid,
        num_sparse_points=num_sparse,
        plot_csv_path=plot_csv_path,
    )

    metrics_json_path = os.path.join(output_dir, "point_cloud_3d_metrics_report.json")
    with open(metrics_json_path, "w") as jf:
        json.dump(summary_metrics, jf, indent=2)

    return PointCloud3DResult(
        las_filepath=las_path,
        ply_filepath=ply_path,
        sparse_ply_filepath=sparse_ply_path,
        dsm_dtm_chm_geotiff=dsm_geotiff_path,
        plot_metrics_csv=plot_csv_path,
        metrics_json_filepath=metrics_json_path,
        dsm_hillshade_png=dsm_hillshade_png,
        chm_map_png=chm_map_png,
        perspective_3d_png=persp_3d_png,
        num_dense_points=num_dense_pts,
        num_sparse_points=num_sparse,
        summary_metrics=summary_metrics,
    )


def _compute_3d_and_plot_metrics(
    grid_w: int,
    grid_h: int,
    origin_e: float,
    origin_n: float,
    cell_size_m: float,
    dsm_grid: array.array,
    dtm_grid: array.array,
    chm_grid: array.array,
    slope_grid: array.array,
    rugosity_grid: array.array,
    ndvi_grid: array.array,
    ndre_grid: array.array,
    tc_grid: array.array,
    zncc_grid: array.array,
    class_grid: array.array,
    num_views_grid: array.array,
    num_sparse_points: int,
    plot_csv_path: str,
) -> Dict[str, Any]:
    """Compute global 3D point cloud statistics and per-plot 3D turfgrass metrics."""
    cell_area_m2 = cell_size_m * cell_size_m
    valid_idx = [i for i in range(grid_w * grid_h) if dsm_grid[i] > -900.0]
    n_valid = len(valid_idx)

    # Sample every 5th point for fast percentile calculation
    dsm_vals = sorted(float(dsm_grid[i]) for i in valid_idx[::5])
    dtm_vals = sorted(float(dtm_grid[i]) for i in valid_idx[::5])
    chm_vals = sorted(float(chm_grid[i]) for i in valid_idx[::5])
    slope_vals = sorted(float(slope_grid[i]) for i in valid_idx[::5])
    zncc_vals = sorted(float(zncc_grid[i]) for i in valid_idx[::5])

    veg_chm_vals = sorted(float(chm_grid[i]) for i in valid_idx if class_grid[i] in (3, 4))

    total_2d_area_m2 = n_valid * cell_area_m2
    total_3d_surface_area_m2 = sum(float(rugosity_grid[i]) for i in valid_idx) * cell_area_m2
    total_canopy_volume_m3 = sum(float(chm_grid[i]) for i in valid_idx if class_grid[i] in (3, 4)) * cell_area_m2

    class_counts = {2: 0, 3: 0, 4: 0, 6: 0}
    for i in valid_idx:
        c = int(class_grid[i])
        if c in class_counts:
            class_counts[c] += 1

    # Partition experimental trial area into 2.0 m x 2.0 m management / evaluation plots
    # Trial core region: Easting [origin_e + 16m .. origin_e + 86m], Northing [origin_n - 39m .. origin_n - 11m]
    plot_size_m = 2.0
    plot_cells = max(1, int(round(plot_size_m / cell_size_m)))
    r_start_px = int(round(11.0 / cell_size_m))
    r_end_px = min(grid_h, int(round(39.0 / cell_size_m)))
    c_start_px = int(round(16.0 / cell_size_m))
    c_end_px = min(grid_w, int(round(86.0 / cell_size_m)))

    plot_rows: List[Dict[str, Any]] = []
    plot_id = 1
    p_row_idx = 1
    for r0 in range(r_start_px, r_end_px - plot_cells + 1, plot_cells):
        p_col_idx = 1
        for c0 in range(c_start_px, c_end_px - plot_cells + 1, plot_cells):
            pts_dsm = []
            pts_dtm = []
            pts_chm = []
            pts_veg_chm = []
            pts_slope = []
            pts_rug = []
            pts_ndvi = []
            pts_ndre = []
            pts_tc = []
            for rr in range(r0, r0 + plot_cells):
                roff = rr * grid_w
                for cc in range(c0, c0 + plot_cells):
                    idx = roff + cc
                    if dsm_grid[idx] > -900.0:
                        pts_dsm.append(float(dsm_grid[idx]))
                        pts_dtm.append(float(dtm_grid[idx]))
                        chm_v = float(chm_grid[idx])
                        pts_chm.append(chm_v)
                        if ndvi_grid[idx] >= 0.24:
                            pts_veg_chm.append(chm_v)
                        pts_slope.append(float(slope_grid[idx]))
                        pts_rug.append(float(rugosity_grid[idx]))
                        pts_ndvi.append(float(ndvi_grid[idx]))
                        pts_ndre.append(float(ndre_grid[idx]))
                        pts_tc.append(float(tc_grid[idx]))

            if len(pts_dsm) >= int(0.8 * plot_cells * plot_cells):
                n_p = len(pts_dsm)
                pts_chm.sort()
                pts_veg_chm.sort()
                fgcc = len(pts_veg_chm) / n_p
                mean_chm_cm = (sum(pts_chm) / n_p) * 100.0
                veg_mean_chm_cm = (sum(pts_veg_chm) / len(pts_veg_chm) * 100.0) if pts_veg_chm else 0.0
                p90_chm_cm = pts_chm[int(0.90 * n_p)] * 100.0
                max_chm_cm = pts_chm[-1] * 100.0
                # Biovolume in cm^3 per m^2 of plot area
                biovol_cm3_m2 = (sum(pts_veg_chm) * cell_area_m2 * 1e6) / (n_p * cell_area_m2)
                center_e = origin_e + (c0 + 0.5 * plot_cells) * cell_size_m
                center_n = origin_n - (r0 + 0.5 * plot_cells) * cell_size_m

                plot_rows.append({
                    "plot_id": f"PLOT_R{p_row_idx:02d}_C{p_col_idx:02d}",
                    "grid_row": p_row_idx,
                    "grid_col": p_col_idx,
                    "center_easting_m": round(center_e, 3),
                    "center_northing_m": round(center_n, 3),
                    "num_3d_points": n_p,
                    "fractional_green_cover": round(fgcc, 4),
                    "mean_dsm_m": round(sum(pts_dsm) / n_p, 4),
                    "mean_dtm_m": round(sum(pts_dtm) / n_p, 4),
                    "mean_canopy_height_cm": round(mean_chm_cm, 2),
                    "vegetated_canopy_height_cm": round(veg_mean_chm_cm, 2),
                    "p90_canopy_height_cm": round(p90_chm_cm, 2),
                    "max_canopy_height_cm": round(max_chm_cm, 2),
                    "canopy_biovolume_cm3_per_m2": round(biovol_cm3_m2, 1),
                    "surface_rugosity_ratio": round(sum(pts_rug) / n_p, 5),
                    "mean_slope_deg": round(sum(pts_slope) / n_p, 2),
                    "mean_ndvi": round(sum(pts_ndvi) / n_p, 4),
                    "mean_ndre": round(sum(pts_ndre) / n_p, 4),
                    "mean_temperature_c": round(sum(pts_tc) / n_p, 2),
                })
                plot_id += 1
            p_col_idx += 1
        p_row_idx += 1

    if plot_rows:
        with open(plot_csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(plot_rows[0].keys()))
            writer.writeheader()
            writer.writerows(plot_rows)

    def _pct(arr: List[float], q: float) -> float:
        if not arr:
            return 0.0
        return round(arr[min(len(arr) - 1, int(len(arr) * q))], 4)

    return {
        "point_cloud_summary": {
            "crs": "EPSG:32614 (WGS 84 / UTM zone 14N)",
            "dense_mvs_points_count": n_valid,
            "sparse_sfm_tie_points_count": num_sparse_points,
            "point_spacing_cm": round(cell_size_m * 100.0, 2),
            "point_density_pts_per_m2": round(1.0 / cell_area_m2, 2),
            "grid_dimensions": [grid_w, grid_h],
            "mean_stereo_views_per_point": round(sum(int(num_views_grid[i]) for i in valid_idx) / max(1, n_valid), 2),
            "mean_mvs_zncc_confidence": round(sum(zncc_vals) / max(1, len(zncc_vals)), 4),
            "median_mvs_zncc_confidence": _pct(zncc_vals, 0.50),
        },
        "3d_surface_and_canopy_metrics": {
            "2d_planar_footprint_area_m2": round(total_2d_area_m2, 2),
            "3d_true_surface_area_m2": round(total_3d_surface_area_m2, 2),
            "overall_surface_rugosity_ratio": round(total_3d_surface_area_m2 / max(1e-6, total_2d_area_m2), 5),
            "total_green_canopy_biovolume_m3": round(total_canopy_volume_m3, 3),
            "dsm_elevation_msl_m": {
                "min": _pct(dsm_vals, 0.0),
                "p05": _pct(dsm_vals, 0.05),
                "median": _pct(dsm_vals, 0.50),
                "mean": round(sum(dsm_vals) / max(1, len(dsm_vals)), 4),
                "p95": _pct(dsm_vals, 0.95),
                "max": _pct(dsm_vals, 1.0),
            },
            "dtm_bare_earth_elevation_msl_m": {
                "min": _pct(dtm_vals, 0.0),
                "p05": _pct(dtm_vals, 0.05),
                "median": _pct(dtm_vals, 0.50),
                "mean": round(sum(dtm_vals) / max(1, len(dtm_vals)), 4),
                "p95": _pct(dtm_vals, 0.95),
                "max": _pct(dtm_vals, 1.0),
            },
            "canopy_height_model_chm_cm": {
                "overall_mean_cm": round((sum(chm_vals) / max(1, len(chm_vals))) * 100.0, 2),
                "vegetated_mean_cm": round((sum(veg_chm_vals) / max(1, len(veg_chm_vals))) * 100.0, 2),
                "vegetated_median_cm": round(_pct(veg_chm_vals, 0.50) * 100.0, 2),
                "vegetated_p90_cm": round(_pct(veg_chm_vals, 0.90) * 100.0, 2),
                "vegetated_p95_cm": round(_pct(veg_chm_vals, 0.95) * 100.0, 2),
                "max_structure_height_cm": round(_pct(chm_vals, 1.0) * 100.0, 2),
            },
            "terrain_slope_deg": {
                "median": _pct(slope_vals, 0.50),
                "mean": round(sum(slope_vals) / max(1, len(slope_vals)), 2),
                "p90": _pct(slope_vals, 0.90),
            },
            "asprs_point_classification_counts": {
                "class_2_bare_ground": class_counts[2],
                "class_3_low_vegetation_turf": class_counts[3],
                "class_4_medium_dense_turf": class_counts[4],
                "class_6_structures_equipment": class_counts[6],
            },
            "segmented_evaluation_plots_count": len(plot_rows),
        },
        "deliverable_files": {
            "las_12_point_cloud_root": "photogrammetric_point_cloud.las",
            "las_12_point_cloud": "output/dense_photogrammetric_point_cloud.las",
            "ply_multi_attribute_point_cloud": "output/dense_photogrammetric_point_cloud.ply",
            "sparse_sfm_tiepoints_ply": "output/sparse_sfm_tiepoints.ply",
            "sparse_sfm_tiepoints_csv": "output/sparse_sfm_tiepoints.csv",
            "dsm_dtm_chm_5band_geotiff": "output/photogrammetric_3d_dsm_dtm_chm.tif",
            "plot_level_3d_metrics_csv": "output/plot_level_3d_metrics.csv",
            "dsm_hillshade_png": "output/point_cloud_dsm_hillshade.png",
            "chm_map_png": "output/point_cloud_chm_map.png",
            "perspective_3d_render_png": "output/point_cloud_3d_perspective_render.png",
        },
    }
