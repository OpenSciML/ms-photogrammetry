"""
ms_photogrammetry: From-Scratch Multispectral UAV Photogrammetry & 3D Point Cloud Package.

Implemented using pure Python standard library modules (`struct`, `array`, `ctypes`, `math`,
`zlib`, `csv`, `json`, `concurrent.futures`) and a custom multithreaded C++17 photogrammetric
core engine (`photogrammetry_engine.cpp`) compiled on demand and bound via `ctypes`.

Public API Summary
------------------
- **Stage 1 (Metadata & Geodesy)**:
    - `inspect_multispectral_dataset`, `infer_utm_zone`, `wgs84_to_utm`, `utm_to_wgs84`,
      `CaptureEvent`, `DatasetInspectionReport`
- **Stage 1b (Radiometric Reflectance & Thermal Calibration)**:
    - `build_radiometric_model`, `calibrate_band_image`, `RadiometricCalibrationModel`
- **Stage 2 (Feature Detection, Undistortion & RANSAC Matching)**:
    - `extract_features_for_capture`, `match_feature_pair`, `FrameFeatures`, `PairwiseMatch`
- **Stage 3 (Structure-from-Motion & Sparse Levenberg-Marquardt Bundle Adjustment)**:
    - `run_sfm_and_bundle_adjustment`, `CameraPose`, `TiePoint3D`, `SfMResult`
- **Stage 4 (Sub-Pixel Multispectral Band-to-Band Registration)**:
    - `estimate_multispectral_registration`, `BandRegistrationParams`, `MultispectralRegistrationModel`
- **Stages 5, 6 & 7 (Continuous DEM Construction, Collinearity Orthorectification & Mosaicking)**:
    - `build_dem_grid`, `generate_multispectral_orthomosaic`, `OrthomosaicProduct`
- **Stage 8 (DEM-Normalized Dense MVS 3D Point Cloud, Bare-Earth DTM, CHM & Plot 3D Metrics)**:
    - `generate_photogrammetric_point_cloud_and_3d_metrics`, `write_las12_point_cloud`,
      `write_ply_point_cloud`, `write_sparse_tiepoints_ply_and_csv`, `PointCloud3DResult`
- **Stage 9 (Quality Assessment & Diagnostics)**:
    - `evaluate_pipeline_quality`, `render_trajectory_and_tiepoints_png`
- **Binary I/O (TIFF 6.0 / EXIF / XMP / GeoTIFF 1.1 / PNG)**:
    - `read_tiff_metadata`, `read_tiff_uint16`, `write_multiband_geotiff`, `write_png_rgb`,
      `read_geotiff_header_summary`, `BandMetadata`
"""

from __future__ import annotations

from .metadata import (
    CaptureEvent,
    DatasetInspectionReport,
    infer_utm_zone,
    inspect_multispectral_dataset,
    utm_to_wgs84,
    wgs84_to_utm,
)
from .radiometry import (
    RadiometricCalibrationModel,
    build_radiometric_model,
    calibrate_band_image,
)
from .features import (
    FrameFeatures,
    PairwiseMatch,
    extract_features_for_capture,
    match_feature_pair,
)
from .sfm import (
    CameraPose,
    SfMResult,
    TiePoint3D,
    run_sfm_and_bundle_adjustment,
)
from .registration import (
    BandRegistrationParams,
    MultispectralRegistrationModel,
    estimate_multispectral_registration,
)
from .orthorectification import (
    OrthomosaicProduct,
    build_dem_grid,
    generate_multispectral_orthomosaic,
)
from .point_cloud import (
    PointCloud3DResult,
    generate_photogrammetric_point_cloud_and_3d_metrics,
    write_las12_point_cloud,
    write_ply_point_cloud,
    write_sparse_tiepoints_ply_and_csv,
)
from .quality import (
    evaluate_pipeline_quality,
    render_trajectory_and_tiepoints_png,
)
from .tiff_io import (
    BandMetadata,
    read_geotiff_header_summary,
    read_tiff_metadata,
    read_tiff_uint16,
    write_multiband_geotiff,
    write_png_rgb,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "BandMetadata",
    "BandRegistrationParams",
    "CameraPose",
    "CaptureEvent",
    "DatasetInspectionReport",
    "FrameFeatures",
    "MultispectralRegistrationModel",
    "OrthomosaicProduct",
    "PairwiseMatch",
    "PointCloud3DResult",
    "RadiometricCalibrationModel",
    "SfMResult",
    "TiePoint3D",
    "build_dem_grid",
    "build_radiometric_model",
    "calibrate_band_image",
    "estimate_multispectral_registration",
    "evaluate_pipeline_quality",
    "extract_features_for_capture",
    "generate_multispectral_orthomosaic",
    "generate_photogrammetric_point_cloud_and_3d_metrics",
    "infer_utm_zone",
    "inspect_multispectral_dataset",
    "match_feature_pair",
    "read_geotiff_header_summary",
    "read_tiff_metadata",
    "read_tiff_uint16",
    "render_trajectory_and_tiepoints_png",
    "run_sfm_and_bundle_adjustment",
    "utm_to_wgs84",
    "wgs84_to_utm",
    "write_las12_point_cloud",
    "write_multiband_geotiff",
    "write_ply_point_cloud",
    "write_png_rgb",
    "write_sparse_tiepoints_ply_and_csv",
]
