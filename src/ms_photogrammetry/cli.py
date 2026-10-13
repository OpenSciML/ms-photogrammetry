"""
Typer Command-Line Interface (CLI) for the From-Scratch Multispectral Photogrammetry Pipeline.

Exposes all input/output folders and photogrammetric parameters directly at the top level of
`ms-photogrammetry` via Typer (https://typer.tiangolo.com/) with Rich-formatted help panels.
"""

from __future__ import annotations
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Annotated, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .metadata import inspect_multispectral_dataset
from .native import get_native_lib
from .orthorectification import build_dem_grid, generate_multispectral_orthomosaic
from .point_cloud import generate_photogrammetric_point_cloud_and_3d_metrics
from .quality import evaluate_pipeline_quality
from .radiometry import build_radiometric_model
from .registration import estimate_multispectral_registration
from .sfm import run_sfm_and_bundle_adjustment

console = Console()

app = typer.Typer(
    name="ms-photogrammetry",
    help=(
        "[bold green]ms-photogrammetry[/bold green]: End-to-end from-scratch multispectral UAV "
        "photogrammetry, DEM/orthomosaic generation, and dense MVS 3D point cloud pipeline "
        "(Pure Python stdlib + Custom C++17 Engine)."
    ),
    rich_markup_mode="rich",
    add_completion=True,
)


def _default_image_dir() -> Path:
    """Resolve the default multispectral imagery directory (`data/Images/multispectral` or `Images/multispectral`)."""
    if os.path.isdir("data/Images/multispectral"):
        return Path("data/Images/multispectral")
    return Path("Images/multispectral")


def _version_callback(value: bool) -> None:
    """Print package version and C++17 engine path when `--version` is passed."""
    if value:
        lib = get_native_lib()
        console.print(
            Panel.fit(
                f"[bold green]ms-photogrammetry[/bold green] v{__version__}\n"
                f"Native C++17 Engine: [cyan]{lib._name}[/cyan]",
                title="Package Info",
            )
        )
        raise typer.Exit()


def execute_pipeline(
    image_dir: Path,
    output_dir: Path,
    resolution_m: float = 0.015,
    point_spacing_m: float = 0.06,
    ba_iterations: int = 12,
    max_captures: int = 0,
    utm_zone: Optional[int] = None,
) -> int:
    """
    Execute the complete 9-stage multispectral photogrammetry and 3D point cloud workflow.

    Args:
        image_dir: Path to input multispectral TIFF directory (`IMG_{capture:04d}_{band}.tif`).
        output_dir: Output directory for GeoTIFFs, LAS/PLY point clouds, PNGs, and JSON/CSV reports.
        resolution_m: Output orthomosaic spatial resolution in meters/pixel (`0.015` = 1.5 cm/px).
        point_spacing_m: Dense MVS 3D point cloud post spacing in meters (`0.06` = 6.0 cm).
        ba_iterations: Number of Levenberg-Marquardt Bundle Adjustment iterations (`12`).
        max_captures: Optional limit on number of survey captures to process (`0` = all).
        utm_zone: Optional WGS84 UTM longitudinal zone (`1..60`). Auto-detected if `None`.

    Returns:
        int: Process exit code (`0` on success).
    """
    out_dir_str = str(output_dir)
    img_dir_str = str(image_dir)
    os.makedirs(out_dir_str, exist_ok=True)
    log_path = os.path.join(out_dir_str, "pipeline_execution.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger("ms_photogrammetry")

    t_start = time.time()
    logger.info("======================================================================")
    logger.info("Starting From-Scratch Multispectral Photogrammetry & 3D Pipeline (v%s)", __version__)
    logger.info("Image Directory : %s", os.path.abspath(img_dir_str))
    logger.info("Output Directory: %s", os.path.abspath(out_dir_str))
    logger.info("Target GSD      : %.4f m/px (%.2f cm/px)", resolution_m, resolution_m * 100.0)
    logger.info("======================================================================")

    # Stage 1: Image Inspection & Metadata Extraction
    t0 = time.time()
    logger.info("[Stage 1/9] Inspecting TIFF headers, EXIF/GPS SubIFDs, XMP calibration, and flight overlap...")
    captures, inspection = inspect_multispectral_dataset(
        image_dir=img_dir_str,
        bands_to_load=(1, 2, 3, 4, 5, 7),
        utm_zone=utm_zone,
    )
    survey_caps = [c for c in captures if c.role == "survey"]
    cal_caps = [c for c in captures if c.role == "calibration_panel"]
    if max_captures > 0:
        survey_caps = survey_caps[:max_captures]

    logger.info(
        "  -> Found %d total files (%d captures): %d survey, %d calibration panels, %d ground/transit.",
        inspection.total_files,
        inspection.total_captures,
        len(survey_caps),
        len(cal_caps),
        len(inspection.transit_captures),
    )
    logger.info(
        "  -> Flight lines: %d | Mean AGL: %.2f m | Native MS GSD: %.2f cm/px | Overlap: %.1f%% fwd, %.1f%% side (%.1fs)",
        len(inspection.flight_lines),
        inspection.mean_agl_m,
        inspection.mean_gsd_ms_m * 100.0,
        inspection.forward_overlap_pct,
        inspection.side_overlap_pct,
        time.time() - t0,
    )

    # Stage 1b: Radiometric Calibration Model
    t0 = time.time()
    logger.info("[Stage 1b/9] Building radiometric reflectance model from RP06 panel captures & DLS...")
    rad_model = build_radiometric_model(cal_caps, bands=(1, 2, 3, 4, 5))
    logger.info(
        "  -> Panel-to-DLS reflectance factors: %s (%.1fs)",
        {k: round(v, 4) for k, v in rad_model.panel_dls_correction.items()},
        time.time() - t0,
    )

    # Stage 2 & 3: Feature Detection, Matching, SfM & Bundle Adjustment
    t0 = time.time()
    logger.info("[Stage 2 & 3/9] Running Shi-Tomasi feature extraction, RANSAC matching, SfM & Bundle Adjustment...")
    sfm_result = run_sfm_and_bundle_adjustment(
        survey_captures=survey_caps,
        ground_elevation_m=inspection.ground_elevation_m,
        rad_model=rad_model,
        ref_band_index=2,
        ba_iterations=ba_iterations,
    )
    logger.info(
        "  -> Verified %d pairwise links (%d captures connected) and triangulated %d 3D tie points.",
        len(sfm_result.pairwise_matches),
        sfm_result.connected_captures,
        len(sfm_result.tie_points),
    )
    logger.info(
        "  -> Reprojection RMSE: initial=%.2f px -> final=%.2f px (check=%.2f px | %.2f cm on ground) (%.1fs)",
        sfm_result.initial_reproj_rmse_px,
        sfm_result.final_reproj_rmse_px,
        sfm_result.check_point_rmse_px,
        sfm_result.final_reproj_rmse_px * inspection.mean_gsd_ms_m * 100.0,
        time.time() - t0,
    )

    # Stage 4: Multispectral Band-to-Band Registration
    t0 = time.time()
    logger.info("[Stage 4/9] Estimating sub-pixel multispectral band registration (Gradient-NCC + Rig Calibration)...")
    reg_model = estimate_multispectral_registration(
        survey_captures=survey_caps,
        mean_agl_m=inspection.mean_agl_m,
        rad_model=rad_model,
        bands_to_register=(1, 2, 3, 4, 5, 7),
    )
    for b_idx, rparam in sorted(reg_model.bands.items()):
        logger.info(
            "  -> Band %d (%-8s): total_shift=(%+6.2f, %+6.2f) px | boresight=(%+6.2f, %+6.2f) px | NCC: %.3f -> %.3f | rig_diff=%.2f px",
            b_idx,
            rparam.band_name,
            rparam.total_du_px,
            rparam.total_dv_px,
            rparam.boresight_du_px,
            rparam.boresight_dv_px,
            rparam.ncc_before,
            rparam.ncc_after,
            rparam.residual_vs_rig_px,
        )

    # Stage 5: Digital Elevation Map (DEM) Construction from Bundle-Adjusted 3D Tie Points
    t0 = time.time()
    logger.info("[Stage 5/9] Computing Digital Elevation Map (DEM) from Bundle-Adjusted 3D tie points...")
    margin_m = 10.5
    eastings = [sfm_result.poses[c.capture_index].easting for c in survey_caps]
    northings = [sfm_result.poses[c.capture_index].northing for c in survey_caps]
    min_e = math.floor((min(eastings) - margin_m) * 10.0) / 10.0
    max_e = math.ceil((max(eastings) + margin_m) * 10.0) / 10.0
    min_n = math.floor((min(northings) - margin_m) * 10.0) / 10.0
    max_n = math.ceil((max(northings) + margin_m) * 10.0) / 10.0
    ortho_w = int(math.ceil((max_e - min_e) / resolution_m))
    ortho_h = int(math.ceil((max_n - min_n) / resolution_m))
    dem_grid, dem_min, dem_max, dem_mean = build_dem_grid(
        sfm_result,
        default_ground_z=inspection.ground_elevation_m,
        ortho_w=ortho_w,
        ortho_h=ortho_h,
        origin_e=min_e,
        origin_n=max_n,
        pixel_size_m=resolution_m,
    )
    logger.info(
        "  -> Computed continuous DEM (%dx%d): min=%.3f m, mean=%.3f m, max=%.3f m MSL (%.1fs)",
        ortho_w,
        ortho_h,
        dem_min,
        dem_mean,
        dem_max,
        time.time() - t0,
    )

    # Stages 6 & 7: Orthorectification onto the DEM, Seamline Mosaicking, and GeoTIFF Export
    t0 = time.time()
    logger.info("[Stages 6 & 7/9] Orthorectifying onto DEM, seamline-blending, and exporting multiband GeoTIFFs...")
    ortho_product = generate_multispectral_orthomosaic(
        survey_captures=survey_caps,
        sfm_result=sfm_result,
        rad_model=rad_model,
        reg_model=reg_model,
        ground_elevation_m=inspection.ground_elevation_m,
        output_dir=out_dir_str,
        pixel_size_m=resolution_m,
        bands_order=(1, 2, 3, 5, 4, 7),
        external_dem_grid=dem_grid,
    )
    logger.info(
        "  -> Generated %dx%d orthomosaic (%.1f m^2 valid coverage, %.1f mean overlapping images/px) in %.1fs",
        ortho_product.width,
        ortho_product.height,
        ortho_product.coverage_area_m2,
        ortho_product.mean_overlap_observations,
        time.time() - t0,
    )

    # Stage 8: Dense Multi-View Stereo (MVS) 3D Point Cloud, Bare-Earth DEM/DTM & 3D Canopy Metrics
    t0 = time.time()
    logger.info("[Stage 8/9] Reconstructing DEM-normalized Dense MVS 3D Point Cloud (LAS 1.2 / PLY) & 3D Metrics...")
    pc_result = generate_photogrammetric_point_cloud_and_3d_metrics(
        survey_captures=survey_caps,
        sfm_result=sfm_result,
        rad_model=rad_model,
        ground_elevation_m=inspection.ground_elevation_m,
        output_dir=out_dir_str,
        ortho_pixel_size_m=resolution_m,
        point_spacing_m=point_spacing_m,
    )
    logger.info(
        "  -> Reconstructed %d DEM-normalized 3D MVS points (%.1fs) -> %s",
        pc_result.num_dense_points,
        time.time() - t0,
        pc_result.las_filepath,
    )

    # Stage 9: Quality Assessment & Diagnostics
    t0 = time.time()
    logger.info("[Stage 9/9] Evaluating geometric/radiometric quality and generating diagnostic plots...")
    evaluate_pipeline_quality(
        inspection=inspection,
        rad_model=rad_model,
        sfm_result=sfm_result,
        reg_model=reg_model,
        ortho_product=ortho_product,
        output_dir=out_dir_str,
    )
    logger.info("  -> Saved quality report to %s/quality_assessment_report.json", out_dir_str)

    elapsed = time.time() - t_start
    table = Table(title=f"Photogrammetric Deliverables Summary ({elapsed:.1f}s)", show_header=True, header_style="bold cyan")
    table.add_column("Artifact", style="bold")
    table.add_column("Path")
    table.add_row("Multiband UInt16 GeoTIFF", ortho_product.multiband_uint16_geotiff)
    table.add_row("Multiband Float32 GeoTIFF", ortho_product.multiband_float32_geotiff)
    table.add_row("NDVI / NDRE GeoTIFF", ortho_product.ndvi_geotiff)
    table.add_row("Digital Elevation Map (DEM)", os.path.join(out_dir_str, "digital_elevation_map_dem.tif"))
    table.add_row("Dense 3D Point Cloud (LAS 1.2)", pc_result.las_filepath)
    table.add_row("Dense 3D Point Cloud (PLY)", pc_result.ply_filepath)
    table.add_row("3D DSM/DTM/CHM GeoTIFF", pc_result.dsm_dtm_chm_geotiff)
    table.add_row("Plot-Level 3D Metrics CSV", pc_result.plot_metrics_csv)
    console.print(table)
    return 0


@app.command()
def cli(
    image_dir: Annotated[
        Path,
        typer.Option(
            "--image-dir",
            "-i",
            help="Path to input multispectral TIFF folder (`IMG_{capture:04d}_{band}.tif`).",
            rich_help_panel="Input & Output Folders",
        ),
    ] = _default_image_dir(),
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for generated GeoTIFF orthomosaics, point clouds, previews, and diagnostic reports.",
            rich_help_panel="Input & Output Folders",
        ),
    ] = Path("output"),
    resolution_m: Annotated[
        float,
        typer.Option(
            "--resolution-m",
            "-r",
            min=0.005,
            max=1.0,
            help="Output orthomosaic spatial resolution in meters/pixel (0.015 = 1.5 cm/px).",
            rich_help_panel="Photogrammetric & 3D Parameters",
        ),
    ] = 0.015,
    point_spacing_m: Annotated[
        float,
        typer.Option(
            "--point-spacing-m",
            "-s",
            min=0.015,
            max=1.0,
            help="Dense MVS 3D point cloud post spacing in meters (0.06 = 6.0 cm).",
            rich_help_panel="Photogrammetric & 3D Parameters",
        ),
    ] = 0.06,
    ba_iterations: Annotated[
        int,
        typer.Option(
            "--ba-iterations",
            "-b",
            min=1,
            max=100,
            help="Number of sparse Levenberg-Marquardt Bundle Adjustment iterations.",
            rich_help_panel="Photogrammetric & 3D Parameters",
        ),
    ] = 12,
    max_captures: Annotated[
        int,
        typer.Option(
            "--max-captures",
            "-n",
            min=0,
            help="Optional limit on number of survey captures to process (0 = all survey captures).",
            rich_help_panel="Photogrammetric & 3D Parameters",
        ),
    ] = 0,
    utm_zone: Annotated[
        Optional[int],
        typer.Option(
            "--utm-zone",
            "-z",
            min=1,
            max=60,
            help="Optional WGS84 UTM longitudinal zone (1..60). Auto-detected from GPS longitude if omitted.",
            rich_help_panel="Geodesy & Inspection",
        ),
    ] = None,
    inspect_only: Annotated[
        bool,
        typer.Option(
            "--inspect-only",
            help="Only inspect dataset metadata, flight lines, GSD, and overlap without running full reconstruction.",
            rich_help_panel="Geodesy & Inspection",
        ),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            is_eager=True,
            help="Display package version and C++17 native engine status and exit.",
        ),
    ] = None,
) -> None:
    """
    Execute the end-to-end multispectral UAV photogrammetry, DEM, orthomosaic, and 3D point cloud pipeline.
    """
    if inspect_only:
        t0 = time.time()
        _, rep = inspect_multispectral_dataset(str(image_dir), utm_zone=utm_zone)
        elapsed = time.time() - t0
        table = Table(
            title=f"UAV Multispectral Dataset Summary ({elapsed:.2f}s)",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Image Directory", os.path.abspath(str(image_dir)))
        table.add_row("Total Files / Captures", f"{rep.total_files:,} files / {rep.total_captures} captures")
        table.add_row(
            "Capture Classification",
            f"{len(rep.survey_captures)} survey | {len(rep.calibration_captures)} calibration panels | {len(rep.transit_captures)} ground/transit",
        )
        table.add_row("Flight Lines Detected", f"{len(rep.flight_lines)} parallel lines")
        table.add_row(
            "Ground / Survey Altitude",
            f"{rep.ground_elevation_m:.2f} m MSL / {rep.mean_survey_altitude_msl_m:.2f} m MSL ({rep.mean_agl_m:.2f} m AGL)",
        )
        table.add_row(
            "Native GSD (MS / Panchro / LWIR)",
            f"{rep.mean_gsd_ms_m*100:.2f} cm / {rep.mean_gsd_panchro_m*100:.2f} cm / {rep.mean_gsd_lwir_m*100:.2f} cm",
        )
        table.add_row("Forward / Side Overlap", f"{rep.forward_overlap_pct:.1f}% forward / {rep.side_overlap_pct:.1f}% side")
        console.print(table)
        return

    code = execute_pipeline(
        image_dir=image_dir,
        output_dir=output_dir,
        resolution_m=resolution_m,
        point_spacing_m=point_spacing_m,
        ba_iterations=ba_iterations,
        max_captures=max_captures,
        utm_zone=utm_zone,
    )
    if code != 0:
        raise typer.Exit(code=code)


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point invoked by `ms-photogrammetry` script or `python -m ms_photogrammetry.cli`.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        app(args=args, prog_name="ms-photogrammetry", standalone_mode=False)
        return 0
    except typer.Exit as exc:
        return int(exc.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
