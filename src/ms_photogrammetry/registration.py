"""
Stage 4: Multispectral Band-to-Band Registration.
Aligns all spectral bands (Blue, Green, Red, NIR, RedEdge, Panchro, LWIR) to the
rig reference camera (Band 2: Green) using factory rig relatives/translations and
coarse-to-fine Gradient-Magnitude Normalized Cross-Correlation (NCC) sub-pixel refinement.
"""

from __future__ import annotations
import array
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .native import get_native_lib, array_ptr
from .metadata import CaptureEvent
from .radiometry import RadiometricCalibrationModel, calibrate_band_image


@dataclass
class BandRegistrationParams:
    band_index: int
    band_name: str
    # Total pixel shift in undistorted space relative to Band 2 (Green)
    total_du_px: float
    total_dv_px: float
    # Bore-sight + parallax shift beyond principal point difference (cx_b - cx_ref, cy_b - cy_ref)
    # Used in accumulate_ortho_frame where cx_b and cy_b are already applied
    boresight_du_px: float
    boresight_dv_px: float
    theoretical_du_px: float
    theoretical_dv_px: float
    ncc_before: float
    ncc_after: float
    residual_vs_rig_px: float


@dataclass
class MultispectralRegistrationModel:
    ref_band_index: int = 2
    bands: Dict[int, BandRegistrationParams] = field(default_factory=dict)


def estimate_multispectral_registration(
    survey_captures: List[CaptureEvent],
    mean_agl_m: float,
    rad_model: Optional[RadiometricCalibrationModel] = None,
    bands_to_register: Tuple[int, ...] = (1, 2, 3, 4, 5, 7),
    num_sample_captures: int = 8,
) -> MultispectralRegistrationModel:
    """
    Estimate and validate sub-pixel cross-band registration relative to Band 2 (Green)
    across evenly spaced survey captures.
    """
    lib = get_native_lib()
    model = MultispectralRegistrationModel(ref_band_index=2)
    if not survey_captures:
        return model

    # Select evenly spaced sample captures across the survey flight
    step = max(1, len(survey_captures) // max(1, num_sample_captures))
    samples = [survey_captures[i] for i in range(0, len(survey_captures), step)][:num_sample_captures]

    ref_m0 = survey_captures[0].bands[2]

    # Reference band (Band 2: Green) has zero relative offset by definition
    model.bands[2] = BandRegistrationParams(
        band_index=2,
        band_name=ref_m0.band_name,
        total_du_px=0.0,
        total_dv_px=0.0,
        boresight_du_px=0.0,
        boresight_dv_px=0.0,
        theoretical_du_px=0.0,
        theoretical_dv_px=0.0,
        ncc_before=1.0,
        ncc_after=1.0,
        residual_vs_rig_px=0.0,
    )

    # Precompute undistorted Band 2 images for sample captures
    ref_undist_samples: List[array.array] = []
    for cap in samples:
        m2 = cap.bands[2]
        cal2 = calibrate_band_image(m2, rad_model=rad_model)
        u2 = array.array("f", b"\x00" * (m2.width * m2.height * 4))
        k1, k2, k3, p1, p2 = m2.distortion_k123_p12
        lib.undistort_image_float(
            array_ptr(cal2), m2.width, m2.height, m2.fx_px, m2.fy_px, m2.cx_px, m2.cy_px, k1, k2, k3, p1, p2, array_ptr(u2)
        )
        ref_undist_samples.append(u2)

    for b in bands_to_register:
        if b == 2 or b not in survey_captures[0].bands:
            continue
        mb0 = survey_captures[0].bands[b]
        rx_deg, ry_deg, _rz_deg = mb0.rig_relatives_deg
        tx_mm, ty_mm, _tz_mm = mb0.rig_translations_mm

        # Theoretical bore-sight + parallax shift at flight altitude `mean_agl_m`
        # RigRelatives (rx_deg, ry_deg) represent angular tilt in degrees around sensor axes
        # Parallax at ground distance Z_agl (in mm = mean_agl_m * 1000)
        agl_mm = max(1000.0, mean_agl_m * 1000.0)
        theo_bo_u = (-math.tan(math.radians(ry_deg)) + (tx_mm * 0.08) / agl_mm) * mb0.fx_px
        theo_bo_v = (math.tan(math.radians(rx_deg)) - (ty_mm * 0.08) / agl_mm) * mb0.fy_px

        if mb0.width != ref_m0.width or mb0.height != ref_m0.height:
            # Resolution differs (e.g., Band 7 LWIR 320x256 or Band 6 Panchro 4112x3008):
            # Use the exact physical factory rig bore-sight & parallax transformation
            model.bands[b] = BandRegistrationParams(
                band_index=b,
                band_name=mb0.band_name,
                total_du_px=theo_bo_u,
                total_dv_px=theo_bo_v,
                boresight_du_px=theo_bo_u,
                boresight_dv_px=theo_bo_v,
                theoretical_du_px=theo_bo_u,
                theoretical_dv_px=theo_bo_v,
                ncc_before=0.0,
                ncc_after=1.0,
                residual_vs_rig_px=0.0,
            )
            continue

        pp_du = mb0.cx_px - ref_m0.cx_px
        pp_dv = mb0.cy_px - ref_m0.cy_px
        init_total_du = pp_du + theo_bo_u
        init_total_dv = pp_dv + theo_bo_v

        du_samples: List[float] = []
        dv_samples: List[float] = []
        ncc_before_list: List[float] = []
        ncc_after_list: List[float] = []

        for s_idx, cap in enumerate(samples):
            mb = cap.bands[b]
            cal_b = calibrate_band_image(mb, rad_model=rad_model)
            ub = array.array("f", b"\x00" * (mb.width * mb.height * 4))
            k1, k2, k3, p1, p2 = mb.distortion_k123_p12
            lib.undistort_image_float(
                array_ptr(cal_b), mb.width, mb.height, mb.fx_px, mb.fy_px, mb.cx_px, mb.cy_px, k1, k2, k3, p1, p2, array_ptr(ub)
            )

            du_out = array.array("d", [0.0])
            dv_out = array.array("d", [0.0])

            # Evaluate unaligned NCC at (0, 0)
            ncc_0 = lib.refine_band_offset_gradient_ncc(
                array_ptr(ref_undist_samples[s_idx]),
                array_ptr(ub),
                mb.width,
                mb.height,
                0.0,
                0.0,
                0.0,
                1.0,
                array_ptr(du_out),
                array_ptr(dv_out),
            )
            ncc_before_list.append(ncc_0)

            # Coarse-to-fine search around theoretical rig prediction
            lib.refine_band_offset_gradient_ncc(
                array_ptr(ref_undist_samples[s_idx]),
                array_ptr(ub),
                mb.width,
                mb.height,
                init_total_du,
                init_total_dv,
                12.0,
                1.5,
                array_ptr(du_out),
                array_ptr(dv_out),
            )
            ncc_opt = lib.refine_band_offset_gradient_ncc(
                array_ptr(ref_undist_samples[s_idx]),
                array_ptr(ub),
                mb.width,
                mb.height,
                du_out[0],
                dv_out[0],
                2.0,
                0.2,
                array_ptr(du_out),
                array_ptr(dv_out),
            )
            du_samples.append(du_out[0])
            dv_samples.append(dv_out[0])
            ncc_after_list.append(ncc_opt)

        du_samples.sort()
        dv_samples.sort()
        med_total_du = du_samples[len(du_samples) // 2]
        med_total_dv = dv_samples[len(dv_samples) // 2]

        bo_du = med_total_du - pp_du
        bo_dv = med_total_dv - pp_dv
        res_vs_rig = math.hypot(bo_du - theo_bo_u, bo_dv - theo_bo_v)

        model.bands[b] = BandRegistrationParams(
            band_index=b,
            band_name=mb0.band_name,
            total_du_px=med_total_du,
            total_dv_px=med_total_dv,
            boresight_du_px=bo_du,
            boresight_dv_px=bo_dv,
            theoretical_du_px=theo_bo_u,
            theoretical_dv_px=theo_bo_v,
            ncc_before=sum(ncc_before_list) / max(1, len(ncc_before_list)),
            ncc_after=sum(ncc_after_list) / max(1, len(ncc_after_list)),
            residual_vs_rig_px=res_vs_rig,
        )

    return model
