"""
Radiometric Calibration Module (Raw 16-bit DN -> Radiance -> Surface Reflectance / Celsius).
Uses calibration panel captures (IMG_0002, IMG_0236) and Downwelling Light Sensor (DLS)
horizontal/spectral irradiance together with the C++17 radiometric engine.
"""

from __future__ import annotations
import array
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .native import get_native_lib, array_ptr
from .tiff_io import BandMetadata, read_tiff_uint16
from .metadata import CaptureEvent


@dataclass
class RadiometricCalibrationModel:
    """Stores per-band empirical panel-to-DLS calibration multipliers."""
    # Multiplier k_band such that: reflectance = L * (pi / (E_dls_W_m2_nm)) * k_band
    panel_dls_correction: Dict[int, float] = field(default_factory=lambda: {b: 1.0 for b in range(1, 8)})
    panel_albedos: Dict[int, float] = field(default_factory=dict)
    panel_radiances: Dict[int, float] = field(default_factory=dict)


def build_radiometric_model(
    calibration_captures: List[CaptureEvent],
    bands: Tuple[int, ...] = (1, 2, 3, 4, 5),
) -> RadiometricCalibrationModel:
    """
    Calibrate the DLS irradiance-to-reflectance conversion against the pre-flight (IMG_0002)
    and post-flight (IMG_0236) reflectance panel captures (RP06-2315088-OB).
    """
    model = RadiometricCalibrationModel()
    if not calibration_captures:
        return model

    lib = get_native_lib()
    for b in bands:
        ratios: List[float] = []
        rads: List[float] = []
        albedos: List[float] = []
        for cap in calibration_captures:
            if b not in cap.bands:
                continue
            m = cap.bands[b]
            if m.panel_albedo is None or m.panel_albedo <= 0:
                continue
            raw = read_tiff_uint16(m)
            p_arr = array.array("i", [p for p, q in m.vignetting_exponents])
            q_arr = array.array("i", [q for p, q in m.vignetting_exponents])
            c_arr = array.array("d", m.vignetting_coeffs)
            out_rad = array.array("f", b"\x00" * (m.width * m.height * 4))
            gain = m.iso_speed / 100.0
            a1, a2, a3 = m.radiometric_cal

            lib.radiometric_calibrate_band(
                array_ptr(raw),
                m.width,
                m.height,
                m.black_level,
                array_ptr(p_arr),
                array_ptr(q_arr),
                array_ptr(c_arr),
                len(c_arr),
                a1,
                a2,
                a3,
                m.exposure_time_s,
                gain,
                m.imager_temp_c,
                1.0,  # output raw spectral radiance W/m^2/sr/nm
                0,
                array_ptr(out_rad),
            )

            # Extract the diffuse Lambertian calibration panel from the central region
            # (the bright square target on the ground represents the 92nd-96th percentile of the central ROI)
            w, h = m.width, m.height
            roi_vals: List[float] = []
            for r in range(h // 4, (3 * h) // 4, 6):
                row_off = r * w
                for c in range(w // 4, (3 * w) // 4, 6):
                    dn_val = raw[row_off + c]
                    if 15000 < dn_val < 62000:
                        roi_vals.append(out_rad[row_off + c])
            if roi_vals:
                roi_vals.sort()
                panel_L = roi_vals[int(len(roi_vals) * 0.90)]
                # DLS SpectralIrradiance is in uW/cm^2/nm -> divide by 100 for W/m^2/nm
                e_dls = (m.spectral_irradiance / 100.0) if m.spectral_irradiance > 0 else 1.0
                dls_refl = panel_L * math.pi / e_dls
                if dls_refl > 0.05:
                    corr = m.panel_albedo / dls_refl
                    # Clamp to physically reasonable panel-DLS cross-calibration factor
                    corr = max(0.55, min(1.35, corr))
                    ratios.append(corr)
                    rads.append(panel_L)
                    albedos.append(m.panel_albedo)

        if ratios:
            model.panel_dls_correction[b] = sum(ratios) / len(ratios)
            model.panel_radiances[b] = sum(rads) / len(rads)
            model.panel_albedos[b] = sum(albedos) / len(albedos)

    return model


def calibrate_band_image(
    meta: BandMetadata,
    rad_model: Optional[RadiometricCalibrationModel] = None,
    raw_dn: Optional[array.array] = None,
) -> array.array:
    """
    Convert a raw 16-bit band image to calibrated Float32 surface reflectance (Bands 1..6)
    or surface temperature in degrees Celsius (Band 7 LWIR).
    """
    lib = get_native_lib()
    if raw_dn is None:
        raw_dn = read_tiff_uint16(meta)

    out = array.array("f", b"\x00" * (meta.width * meta.height * 4))
    is_thermal = 1 if (meta.band_index == 7 or meta.band_name.upper() == "LWIR") else 0

    if is_thermal:
        lib.radiometric_calibrate_band(
            array_ptr(raw_dn),
            meta.width,
            meta.height,
            0.0,
            0,
            0,
            0,
            0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1,
            array_ptr(out),
        )
        return out

    p_arr = array.array("i", [p for p, q in meta.vignetting_exponents])
    q_arr = array.array("i", [q for p, q in meta.vignetting_exponents])
    c_arr = array.array("d", meta.vignetting_coeffs)
    gain = meta.iso_speed / 100.0
    a1, a2, a3 = meta.radiometric_cal

    # Convert DLS irradiance (uW/cm^2/nm -> W/m^2/nm by dividing by 100)
    irr_uw = meta.horizontal_irradiance if meta.horizontal_irradiance > 10.0 else meta.spectral_irradiance
    irr_w = (irr_uw / 100.0) if irr_uw > 0 else 1.2
    panel_corr = rad_model.panel_dls_correction.get(meta.band_index, 1.0) if rad_model else 1.0
    radiance_to_refl = (math.pi / irr_w) * panel_corr

    lib.radiometric_calibrate_band(
        array_ptr(raw_dn),
        meta.width,
        meta.height,
        meta.black_level,
        array_ptr(p_arr),
        array_ptr(q_arr),
        array_ptr(c_arr),
        len(c_arr),
        a1,
        a2,
        a3,
        meta.exposure_time_s,
        gain,
        meta.imager_temp_c,
        radiance_to_refl,
        0,
        array_ptr(out),
    )
    return out
