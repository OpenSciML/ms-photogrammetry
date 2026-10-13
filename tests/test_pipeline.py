"""
Unit and stage-validation tests for the from-scratch multispectral photogrammetry pipeline.
Run with: uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations
import array
import math
import os
import tempfile
import unittest

from ms_photogrammetry.native import get_native_lib, array_ptr
from ms_photogrammetry.metadata import wgs84_to_utm, utm_to_wgs84, CaptureEvent
from ms_photogrammetry.tiff_io import (
    read_tiff_metadata,
    read_tiff_uint16,
    write_multiband_geotiff,
    read_geotiff_header_summary,
)
from ms_photogrammetry.radiometry import calibrate_band_image
from ms_photogrammetry.features import extract_features_for_capture, match_feature_pair
from ms_photogrammetry.sfm import CameraPose, triangulate_track_3d, project_point_undistorted


IMAGE_DIR = "data/Images/multispectral" if os.path.isdir("data/Images/multispectral") else "Images/multispectral"


class TestPhotogrammetryPipeline(unittest.TestCase):
    def test_wgs84_utm_roundtrip(self) -> None:
        lat0, lon0 = 32.986138, -96.763841
        e, n = wgs84_to_utm(lat0, lon0, zone=14)
        self.assertTrue(700000.0 < e < 720000.0)
        self.assertTrue(3650000.0 < n < 3660000.0)
        lat1, lon1 = utm_to_wgs84(e, n, zone=14, northern=True)
        self.assertAlmostEqual(lat0, lat1, places=7)
        self.assertAlmostEqual(lon0, lon1, places=7)

    def test_tiff_metadata_and_pixel_reading(self) -> None:
        p = os.path.join(IMAGE_DIR, "IMG_0000_1.tif")
        meta = read_tiff_metadata(p, capture_index=0, band_index=1)
        self.assertEqual(meta.width, 2064)
        self.assertEqual(meta.height, 1544)
        self.assertEqual(meta.band_name, "Blue")
        self.assertAlmostEqual(meta.central_wavelength_nm, 475.0)
        self.assertGreater(meta.fx_px, 2200.0)
        self.assertAlmostEqual(meta.latitude_deg, 32.986094, places=4)
        self.assertAlmostEqual(meta.longitude_deg, -96.763992, places=4)

        raw = read_tiff_uint16(meta)
        self.assertEqual(len(raw), 2064 * 1544)
        self.assertGreater(max(raw), 4000)

    def test_brown_conrady_undistortion_inversion(self) -> None:
        lib = get_native_lib()
        fx, fy, cx, cy = 2283.5, 2283.5, 1035.0, 758.0
        k1, k2, k3, p1, p2 = -0.1329, 0.1569, -0.0403, -5.2e-6, 7.04e-4

        # Start with known undistorted point (u_true, v_true)
        u_true, v_true = 1420.0, 380.0
        xn = (u_true - cx) / fx
        yn = (v_true - cy) / fy
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        ud = fx * xd + cx
        vd = fy * yd + cy

        in_uv = array.array("d", [ud, vd])
        out_uv = array.array("d", [0.0, 0.0])
        lib.undistort_points_2d(
            array_ptr(in_uv), 1, fx, fy, cx, cy, k1, k2, k3, p1, p2, array_ptr(out_uv)
        )
        self.assertAlmostEqual(out_uv[0], u_true, places=4)
        self.assertAlmostEqual(out_uv[1], v_true, places=4)

    def test_radiometric_calibration(self) -> None:
        m_red = read_tiff_metadata(os.path.join(IMAGE_DIR, "IMG_0050_3.tif"), 50, 3)
        m_nir = read_tiff_metadata(os.path.join(IMAGE_DIR, "IMG_0050_4.tif"), 50, 4)
        m_lwir = read_tiff_metadata(os.path.join(IMAGE_DIR, "IMG_0050_7.tif"), 50, 7)

        cal_red = calibrate_band_image(m_red)
        cal_nir = calibrate_band_image(m_nir)
        cal_lwir = calibrate_band_image(m_lwir)

        mid_idx = (m_red.height // 2) * m_red.width + (m_red.width // 2)
        r_val = cal_red[mid_idx]
        n_val = cal_nir[mid_idx]
        ndvi = (n_val - r_val) / (n_val + r_val)

        # Healthy turfgrass has Red reflectance < NIR reflectance and positive NDVI
        self.assertTrue(0.01 < r_val < 0.20)
        self.assertTrue(0.15 < n_val < 0.70)
        self.assertTrue(0.35 < ndvi < 0.92)
        # LWIR thermal in degrees Celsius
        t_c = cal_lwir[(m_lwir.height // 2) * m_lwir.width + (m_lwir.width // 2)]
        self.assertTrue(10.0 < t_c < 55.0)

    def test_camera_rotation_so3_and_triangulation(self) -> None:
        pose1 = CameraPose(10, 708800.0, 3652100.0, 231.0, -0.12, 0.01, -0.01, 708800.0, 3652100.0, 231.0, -0.12, 0.0, 0.0)
        pose2 = CameraPose(11, 708803.0, 3652100.2, 231.0, -0.12, 0.01, -0.01, 708803.0, 3652100.2, 231.0, -0.12, 0.0, 0.0)
        R1 = pose1.rotation_matrix_world_to_cam()
        R2 = pose2.rotation_matrix_world_to_cam()

        # Verify orthogonality R * R^T = I and det(R) = +1
        det = (
            R1[0] * (R1[4] * R1[8] - R1[5] * R1[7])
            - R1[1] * (R1[3] * R1[8] - R1[5] * R1[6])
            + R1[2] * (R1[3] * R1[7] - R1[4] * R1[6])
        )
        self.assertAlmostEqual(det, 1.0, places=6)

        fx, fy, cx, cy = 2283.5, 2283.5, 1032.0, 772.0
        true_pt = (708801.5, 3652101.0, 211.2)
        uv1 = project_point_undistorted(pose1, R1, *true_pt, fx, fy, cx, cy)
        uv2 = project_point_undistorted(pose2, R2, *true_pt, fx, fy, cx, cy)
        self.assertIsNotNone(uv1)
        self.assertIsNotNone(uv2)

        tri = triangulate_track_3d(
            [(10, uv1[0], uv1[1]), (11, uv2[0], uv2[1])],
            {10: pose1, 11: pose2},
            {10: R1, 11: R2},
            fx, fy, cx, cy,
            prior_ground_z=211.2,
        )
        self.assertIsNotNone(tri)
        self.assertAlmostEqual(tri[0], true_pt[0], places=2)
        self.assertAlmostEqual(tri[1], true_pt[1], places=2)
        self.assertAlmostEqual(tri[2], true_pt[2], places=2)

    def test_feature_detection_and_ransac_matching(self) -> None:
        m10 = read_tiff_metadata(os.path.join(IMAGE_DIR, "IMG_0010_1.tif"), 10, 1)
        m11 = read_tiff_metadata(os.path.join(IMAGE_DIR, "IMG_0011_1.tif"), 11, 1)
        e10, n10 = wgs84_to_utm(m10.latitude_deg, m10.longitude_deg)
        e11, n11 = wgs84_to_utm(m11.latitude_deg, m11.longitude_deg)
        c10 = CaptureEvent(10, m10.capture_id, m10.datetime_str, m10.latitude_deg, m10.longitude_deg, m10.altitude_m, e10, n10, m10.dls_yaw_rad, m10.dls_pitch_rad, m10.dls_roll_rad, m10.trigger_method, False, "survey", bands={1: m10})
        c11 = CaptureEvent(11, m11.capture_id, m11.datetime_str, m11.latitude_deg, m11.longitude_deg, m11.altitude_m, e11, n11, m11.dls_yaw_rad, m11.dls_pitch_rad, m11.dls_roll_rad, m11.trigger_method, False, "survey", bands={1: m11})

        f10 = extract_features_for_capture(c10, band_index=1, max_features=500)
        f11 = extract_features_for_capture(c11, band_index=1, max_features=500)
        self.assertGreater(f10.num_keypoints, 200)
        self.assertGreater(f11.num_keypoints, 200)

        match = match_feature_pair(f10, f11, search_radius_px=0.0)
        self.assertIsNotNone(match)
        self.assertGreater(match.num_inliers, 40)
        self.assertLess(match.inlier_rmse_px, 5.0)

    def test_multiband_geotiff_writer_and_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tif = os.path.join(tmpdir, "test_ortho.tif")
            w, h = 64, 48
            b1 = array.array("H", [1200] * (w * h)).tobytes()
            b2 = array.array("H", [3400] * (w * h)).tobytes()
            write_multiband_geotiff(
                filepath=out_tif,
                bands_data=[b1, b2],
                width=w,
                height=h,
                origin_easting=708800.0,
                origin_northing=3652150.0,
                pixel_size_x=0.015,
                pixel_size_y=0.015,
                band_descriptions=[
                    {"name": "Blue", "central_wavelength_nm": 475, "fwhm_nm": 32, "units": "reflectance"},
                    {"name": "Green", "central_wavelength_nm": 560, "fwhm_nm": 27, "units": "reflectance"},
                ],
                sample_format="uint16",
                epsg_code=32614,
            )
            summary = read_geotiff_header_summary(out_tif)
            self.assertEqual(summary["magic"], 42)
            self.assertEqual(summary["width"], w)
            self.assertEqual(summary["height"], h)
            self.assertEqual(summary["samples_per_pixel"], 2)
            self.assertEqual(summary["planar_configuration"], 2)
            self.assertAlmostEqual(summary["pixel_scale"][0], 0.015)
            self.assertAlmostEqual(summary["tiepoint"][3], 708800.0)
            self.assertAlmostEqual(summary["tiepoint"][4], 3652150.0)
            self.assertIn(32614, summary["geokeys"])

    def test_las12_and_ply_point_cloud_writers(self) -> None:
        from ms_photogrammetry.point_cloud import write_las12_point_cloud, write_ply_point_cloud
        import struct

        with tempfile.TemporaryDirectory() as tmpdir:
            las_file = os.path.join(tmpdir, "test_cloud.las")
            ply_file = os.path.join(tmpdir, "test_cloud.ply")
            gw, gh = 16, 12
            n_tot = gw * gh
            dsm = array.array("f", [211.85] * n_tot)
            chm = array.array("f", [0.045] * n_tot)
            nir = array.array("f", [0.32] * n_tot)
            ndvi = array.array("f", [0.68] * n_tot)
            ndre = array.array("f", [0.42] * n_tot)
            tc = array.array("f", [38.5] * n_tot)
            zncc = array.array("f", [0.81] * n_tot)
            normals = array.array("f", [0.0, 0.0, 1.0] * n_tot)
            rgb = array.array("B", [60, 160, 50, 255] * n_tot)
            cls_arr = array.array("B", [3] * n_tot)
            nviews = array.array("H", [12] * n_tot)

            n_las = write_las12_point_cloud(
                las_file, gw, gh, 708922.7, 3652009.3, 0.06, dsm, nir, rgb, cls_arr, nviews
            )
            self.assertEqual(n_las, n_tot)
            self.assertEqual(os.path.getsize(las_file), 227 + n_tot * 26)
            with open(las_file, "rb") as f:
                sig = f.read(4)
                self.assertEqual(sig, b"LASF")
                f.seek(24)
                ver_maj, ver_min = struct.unpack("<BB", f.read(2))
                self.assertEqual((ver_maj, ver_min), (1, 2))

            n_ply = write_ply_point_cloud(
                ply_file, gw, gh, 708922.7, 3652009.3, 0.06, dsm, chm, normals, rgb, ndvi, ndre, tc, zncc, cls_arr
            )
            self.assertEqual(n_ply, n_tot)
            self.assertGreater(os.path.getsize(ply_file), n_tot * 44)


if __name__ == "__main__":
    unittest.main()

