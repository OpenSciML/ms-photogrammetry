"""
Stage 1: Image Inspection, WGS84/UTM Geodesy, Trajectory Classification, and Overlap Analysis.
Implemented from scratch using Python's standard library.
"""

from __future__ import annotations
import glob
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .tiff_io import BandMetadata, read_tiff_metadata


# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)
UTM_K0 = 0.9996


def infer_utm_zone(lon_deg: float) -> int:
    """
    Infer the standard WGS84 UTM longitudinal zone (1..60) from geodetic longitude in degrees.

    Args:
        lon_deg: Geodetic longitude in decimal degrees (-180.0 to +180.0).

    Returns:
        int: UTM zone number (e.g., 14 for Dallas, TX at ~-96.76 deg).
    """
    return max(1, min(60, int((lon_deg + 180.0) // 6.0) + 1))


def wgs84_to_utm(lat_deg: float, lon_deg: float, zone: Optional[int] = None) -> Tuple[float, float]:
    """
    Convert WGS84 geodetic coordinates (latitude, longitude in degrees) to
    UTM Easting and Northing (meters) using Krüger's 6th-order Transverse Mercator expansion.

    Args:
        lat_deg: Geodetic latitude in decimal degrees.
        lon_deg: Geodetic longitude in decimal degrees.
        zone: Optional UTM zone (1..60). If None, inferred automatically via `infer_utm_zone(lon_deg)`.

    Returns:
        Tuple[float, float]: `(easting_m, northing_m)` in meters.
    """
    if zone is None:
        zone = infer_utm_zone(lon_deg)
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)  # Central meridian (-99.0 deg for Zone 14)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    tan_phi = math.tan(phi)

    n_val = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
    t_val = tan_phi * tan_phi
    c_val = WGS84_EP2 * cos_phi * cos_phi
    a_val = cos_phi * (lam - lam0)

    e2 = WGS84_E2
    e4 = e2 * e2
    e6 = e4 * e2

    m_val = WGS84_A * (
        (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * phi
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * math.sin(2.0 * phi)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * math.sin(4.0 * phi)
        - (35.0 * e6 / 3072.0) * math.sin(6.0 * phi)
    )

    a2 = a_val * a_val
    a3 = a2 * a_val
    a4 = a3 * a_val
    a5 = a4 * a_val
    a6 = a5 * a_val

    easting = 500000.0 + UTM_K0 * n_val * (
        a_val
        + (1.0 - t_val + c_val) * a3 / 6.0
        + (5.0 - 18.0 * t_val + t_val * t_val + 72.0 * c_val - 58.0 * WGS84_EP2) * a5 / 120.0
    )

    northing = UTM_K0 * (
        m_val
        + n_val
        * tan_phi
        * (
            a2 / 2.0
            + (5.0 - t_val + 9.0 * c_val + 4.0 * c_val * c_val) * a4 / 24.0
            + (61.0 - 58.0 * t_val + t_val * t_val + 600.0 * c_val - 330.0 * WGS84_EP2) * a6 / 720.0
        )
    )
    if lat_deg < 0:
        northing += 10000000.0
    return easting, northing


def utm_to_wgs84(easting: float, northing: float, zone: int = 14, northern: bool = True) -> Tuple[float, float]:
    """
    Inverse Transverse Mercator: convert UTM `(easting, northing)` in meters to
    WGS84 geodetic `(latitude_deg, longitude_deg)` in decimal degrees.

    Args:
        easting: UTM Easting in meters.
        northing: UTM Northing in meters.
        zone: UTM longitudinal zone (1..60, default 14).
        northern: True for Northern Hemisphere (`EPSG:326xx`), False for Southern (`EPSG:327xx`).

    Returns:
        Tuple[float, float]: `(latitude_deg, longitude_deg)`.
    """
    x = easting - 500000.0
    y = northing if northern else (northing - 10000000.0)
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)

    e2 = WGS84_E2
    e4 = e2 * e2
    e6 = e4 * e2
    m_val = y / UTM_K0
    mu = m_val / (WGS84_A * (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0))

    e1 = (1.0 - math.sqrt(1.0 - e2)) / (1.0 + math.sqrt(1.0 - e2))
    e1_2 = e1 * e1
    e1_3 = e1_2 * e1
    e1_4 = e1_3 * e1

    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1_3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1_2 / 16.0 - 55.0 * e1_4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1_3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1_4 / 512.0) * math.sin(8.0 * mu)
    )

    sin_p1 = math.sin(phi1)
    cos_p1 = math.cos(phi1)
    tan_p1 = math.tan(phi1)

    n1 = WGS84_A / math.sqrt(1.0 - e2 * sin_p1 * sin_p1)
    t1 = tan_p1 * tan_p1
    c1 = WGS84_EP2 * cos_p1 * cos_p1
    r1 = WGS84_A * (1.0 - e2) / math.pow(1.0 - e2 * sin_p1 * sin_p1, 1.5)
    d = x / (n1 * UTM_K0)
    d2 = d * d
    d3 = d2 * d
    d4 = d3 * d
    d5 = d4 * d
    d6 = d5 * d

    lat_rad = phi1 - (n1 * tan_p1 / r1) * (
        d2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * WGS84_EP2) * d4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 * t1 - 252.0 * WGS84_EP2 - 3.0 * c1 * c1) * d6 / 720.0
    )
    lon_rad = lam0 + (
        d
        - (1.0 + 2.0 * t1 + c1) * d3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 + 8.0 * WGS84_EP2 + 24.0 * t1 * t1) * d5 / 120.0
    ) / cos_p1

    return math.degrees(lat_rad), math.degrees(lon_rad)


@dataclass
class CaptureEvent:
    """Represents a single multi-band UAV capture event with geodetic/UTM coordinates and attitude."""
    capture_index: int
    capture_id: str
    datetime_str: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    easting_m: float
    northing_m: float
    dls_yaw_rad: float
    dls_pitch_rad: float
    dls_roll_rad: float
    trigger_method: int
    is_calibration_panel: bool
    role: str  # "calibration_panel", "survey", "ground_or_transit"
    flight_line_id: int = -1
    bands: Dict[int, BandMetadata] = field(default_factory=dict)

    def band_filepath(self, image_dir: str, band_idx: int) -> str:
        """Return the file path for a given spectral band index (`IMG_{idx:04d}_{band}.tif`)."""
        return os.path.join(image_dir, f"IMG_{self.capture_index:04d}_{band_idx}.tif")


@dataclass
class DatasetInspectionReport:
    """Summary report of Stage 1 dataset inspection, sensor intrinsics, flight lines, and overlap."""
    image_dir: str
    total_files: int
    total_captures: int
    survey_captures: List[int]
    calibration_captures: List[int]
    transit_captures: List[int]
    ground_elevation_m: float
    mean_survey_altitude_msl_m: float
    mean_agl_m: float
    mean_gsd_ms_m: float
    mean_gsd_panchro_m: float
    mean_gsd_lwir_m: float
    footprint_ms_m: Tuple[float, float]
    mean_forward_step_m: float
    mean_line_spacing_m: float
    forward_overlap_pct: float
    side_overlap_pct: float
    flight_lines: List[List[int]]
    band_specs: Dict[int, Dict[str, Any]]
    utm_bounds_survey: Tuple[float, float, float, float]  # (min_e, min_n, max_e, max_n)


def inspect_multispectral_dataset(
    image_dir: str,
    bands_to_load: Tuple[int, ...] = (1, 2, 3, 4, 5, 7),
    survey_min_alt_m: Optional[float] = None,
    utm_zone: Optional[int] = None,
) -> Tuple[List[CaptureEvent], DatasetInspectionReport]:
    """
    Stage 1: Scan all captures in `image_dir`, parse binary TIFF/EXIF/GPS/XMP metadata,
    classify captures into calibration panels / transit / survey flight lines,
    and compute sensor intrinsics, ground footprints, and image overlap.

    Args:
        image_dir: Directory containing `IMG_{capture:04d}_{band}.tif` files.
        bands_to_load: Tuple of 1-based band indices to parse for active captures.
        survey_min_alt_m: Optional minimum altitude (m MSL) to classify a frame as a survey capture.
            If None, automatically inferred from the dataset's ground-to-cruise altitude distribution.
        utm_zone: Optional WGS84 UTM zone (1..60). If None, inferred automatically from GPS longitude.

    Returns:
        Tuple[List[CaptureEvent], DatasetInspectionReport]: Ordered list of `CaptureEvent` objects
        and the summary `DatasetInspectionReport`.
    """
    band1_files = sorted(glob.glob(os.path.join(image_dir, "IMG_*_1.tif")))
    all_files = glob.glob(os.path.join(image_dir, "IMG_*.tif"))
    if not band1_files:
        raise FileNotFoundError(f"No IMG_*_1.tif files found in {image_dir}")

    capture_indices: List[int] = []
    for p in band1_files:
        base = os.path.basename(p)
        idx = int(base.split("_")[1])
        capture_indices.append(idx)

    # Read full 7-band metadata for capture 0000 and panel captures 0002 & 0236
    # and Band 1 metadata for all captures in parallel
    def _load_b1(idx: int) -> BandMetadata:
        p = os.path.join(image_dir, f"IMG_{idx:04d}_1.tif")
        return read_tiff_metadata(p, capture_index=idx, band_index=1)

    with ThreadPoolExecutor(max_workers=16) as ex:
        b1_metas = list(ex.map(_load_b1, capture_indices))

    # Load all 7 bands of IMG_0000 for reference camera intrinsics & rig specs
    ref_bands_0000: Dict[int, BandMetadata] = {}
    for b in range(1, 8):
        p0 = os.path.join(image_dir, f"IMG_0000_{b}.tif")
        if os.path.exists(p0):
            ref_bands_0000[b] = read_tiff_metadata(p0, capture_index=0, band_index=b)

    # Auto-detect UTM zone and survey altitude threshold if not explicitly specified
    if utm_zone is None and b1_metas:
        utm_zone = infer_utm_zone(b1_metas[0].longitude_deg)

    alts_sorted = sorted(m.altitude_m for m in b1_metas)
    min_alt_obs = alts_sorted[0] if alts_sorted else 0.0
    max_alt_obs = alts_sorted[-1] if alts_sorted else 0.0
    if survey_min_alt_m is None:
        # Automatically separate ground/takeoff captures from cruise survey captures
        survey_min_alt_m = min_alt_obs + 0.40 * max(10.0, max_alt_obs - min_alt_obs)

    captures: List[CaptureEvent] = []
    ground_alts: List[float] = []

    for m1 in b1_metas:
        east, north = wgs84_to_utm(m1.latitude_deg, m1.longitude_deg, zone=utm_zone)
        is_panel = (m1.trigger_method == 1) or (m1.panel_albedo is not None)
        if is_panel:
            role = "calibration_panel"
            ground_alts.append(m1.altitude_m)
        elif m1.altitude_m >= survey_min_alt_m:
            role = "survey"
        else:
            role = "ground_or_transit"
            if m1.altitude_m <= min_alt_obs + 2.5:
                ground_alts.append(m1.altitude_m)

        cap = CaptureEvent(
            capture_index=m1.capture_index,
            capture_id=m1.capture_id,
            datetime_str=m1.datetime_str,
            latitude_deg=m1.latitude_deg,
            longitude_deg=m1.longitude_deg,
            altitude_m=m1.altitude_m,
            easting_m=east,
            northing_m=north,
            dls_yaw_rad=m1.dls_yaw_rad,
            dls_pitch_rad=m1.dls_pitch_rad,
            dls_roll_rad=m1.dls_roll_rad,
            trigger_method=m1.trigger_method,
            is_calibration_panel=is_panel,
            role=role,
            bands={1: m1},
        )
        captures.append(cap)

    # Load all requested bands for calibration_panel and survey captures in parallel
    active_caps = [c for c in captures if c.role in ("calibration_panel", "survey")]
    tasks_to_load: List[Tuple[CaptureEvent, int]] = []
    for c in active_caps:
        for b in bands_to_load:
            if b != 1:
                tasks_to_load.append((c, b))

    def _load_other_band(item: Tuple[CaptureEvent, int]) -> Tuple[int, int, BandMetadata]:
        c, b = item
        p = c.band_filepath(image_dir, b)
        return c.capture_index, b, read_tiff_metadata(p, capture_index=c.capture_index, band_index=b)

    cap_by_idx = {c.capture_index: c for c in captures}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for c_idx, b_idx, b_meta in ex.map(_load_other_band, tasks_to_load):
            cap_by_idx[c_idx].bands[b_idx] = b_meta

    # Segment survey captures into flight lines based on Easting direction reversals / Northing jumps
    survey_caps = [c for c in captures if c.role == "survey"]
    flight_lines: List[List[int]] = []
    if survey_caps:
        cur_line = [survey_caps[0].capture_index]
        survey_caps[0].flight_line_id = 0
        prev_de = 0.0
        for k in range(1, len(survey_caps)):
            c_prev = survey_caps[k - 1]
            c_cur = survey_caps[k]
            de = c_cur.easting_m - c_prev.easting_m
            dn = abs(c_cur.northing_m - c_prev.northing_m)
            # Detect flight line turn: large Northing step (> 1.8m) or reversal of East-West velocity
            if dn > 1.8 or (prev_de * de < 0 and abs(de) > 0.5 and len(cur_line) >= 5):
                flight_lines.append(cur_line)
                cur_line = [c_cur.capture_index]
                c_cur.flight_line_id = len(flight_lines)
                prev_de = 0.0
            else:
                cur_line.append(c_cur.capture_index)
                c_cur.flight_line_id = len(flight_lines)
                if abs(de) > 0.5:
                    prev_de = de
        if cur_line:
            flight_lines.append(cur_line)

    ground_elev = sum(ground_alts) / len(ground_alts) if ground_alts else 211.15
    survey_alts = [c.altitude_m for c in survey_caps]
    mean_survey_alt = sum(survey_alts) / len(survey_alts) if survey_alts else 230.9
    mean_agl = mean_survey_alt - ground_elev

    # Compute GSD and footprint for MS (Band 2 Green reference), Panchro (Band 6), LWIR (Band 7)
    b2_ref = ref_bands_0000.get(2, b1_metas[0])
    gsd_ms = mean_agl / b2_ref.fx_px
    footprint_w = b2_ref.width * gsd_ms
    footprint_h = b2_ref.height * gsd_ms

    b6_ref = ref_bands_0000.get(6)
    gsd_pan = (mean_agl / b6_ref.fx_px) if b6_ref else gsd_ms * 0.47
    b7_ref = ref_bands_0000.get(7)
    gsd_lwir = (mean_agl / b7_ref.fx_px) if b7_ref else gsd_ms * 6.1

    # Compute forward spacing along flight lines and side spacing between consecutive flight lines
    forward_steps: List[float] = []
    line_northing_means: List[float] = []
    for fl in flight_lines:
        if len(fl) >= 3:
            line_northing_means.append(sum(cap_by_idx[i].northing_m for i in fl) / len(fl))
            for a_i, b_i in zip(fl[:-1], fl[1:]):
                ca, cb = cap_by_idx[a_i], cap_by_idx[b_i]
                dist = math.hypot(cb.easting_m - ca.easting_m, cb.northing_m - ca.northing_m)
                if 0.3 <= dist <= 6.0:
                    forward_steps.append(dist)

    line_spacings: List[float] = []
    for n1, n2 in zip(line_northing_means[:-1], line_northing_means[1:]):
        dn = abs(n2 - n1)
        if 1.0 <= dn <= 10.0:
            line_spacings.append(dn)

    mean_fwd_step = sum(forward_steps) / len(forward_steps) if forward_steps else 2.35
    mean_side_step = sum(line_spacings) / len(line_spacings) if line_spacings else 2.85

    # Drone flies East-West with camera top/bottom or left/right aligned to flight track
    # Footprint along-track and cross-track overlap:
    fwd_overlap_pct = max(0.0, (1.0 - mean_fwd_step / footprint_w) * 100.0)
    side_overlap_pct = max(0.0, (1.0 - mean_side_step / footprint_h) * 100.0)

    band_specs: Dict[int, Dict[str, Any]] = {}
    for b_idx, bm in sorted(ref_bands_0000.items()):
        band_specs[b_idx] = {
            "band_index": b_idx,
            "name": bm.band_name,
            "central_wavelength_nm": bm.central_wavelength_nm,
            "fwhm_nm": bm.wavelength_fwhm_nm,
            "width": bm.width,
            "height": bm.height,
            "focal_length_mm": bm.focal_length_mm,
            "fx_px": round(bm.fx_px, 2),
            "fy_px": round(bm.fy_px, 2),
            "cx_px": round(bm.cx_px, 2),
            "cy_px": round(bm.cy_px, 2),
            "distortion_k123_p12": bm.distortion_k123_p12,
            "rig_relatives_deg": bm.rig_relatives_deg,
            "rig_translations_mm": bm.rig_translations_mm,
        }

    eastings = [c.easting_m for c in survey_caps]
    northings = [c.northing_m for c in survey_caps]
    utm_bounds = (min(eastings), min(northings), max(eastings), max(northings))

    report = DatasetInspectionReport(
        image_dir=image_dir,
        total_files=len(all_files),
        total_captures=len(captures),
        survey_captures=[c.capture_index for c in survey_caps],
        calibration_captures=[c.capture_index for c in captures if c.role == "calibration_panel"],
        transit_captures=[c.capture_index for c in captures if c.role == "ground_or_transit"],
        ground_elevation_m=ground_elev,
        mean_survey_altitude_msl_m=mean_survey_alt,
        mean_agl_m=mean_agl,
        mean_gsd_ms_m=gsd_ms,
        mean_gsd_panchro_m=gsd_pan,
        mean_gsd_lwir_m=gsd_lwir,
        footprint_ms_m=(footprint_w, footprint_h),
        mean_forward_step_m=mean_fwd_step,
        mean_line_spacing_m=mean_side_step,
        forward_overlap_pct=fwd_overlap_pct,
        side_overlap_pct=side_overlap_pct,
        flight_lines=flight_lines,
        band_specs=band_specs,
        utm_bounds_survey=utm_bounds,
    )
    return captures, report
