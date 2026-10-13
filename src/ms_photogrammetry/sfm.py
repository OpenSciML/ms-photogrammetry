"""
Stage 3: Structure-from-Motion (SfM), Multi-View Track Triangulation,
Global Pose Graph Optimization, and Sparse Levenberg-Marquardt Bundle Adjustment.
Implemented from scratch using Python's standard library.
"""

from __future__ import annotations
import array
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .metadata import CaptureEvent
from .radiometry import RadiometricCalibrationModel
from .features import FrameFeatures, PairwiseMatch, extract_features_for_capture, match_feature_pair


@dataclass
class CameraPose:
    capture_index: int
    # Optimized exterior orientation in UTM Zone 14N (meters) and radians
    easting: float
    northing: float
    altitude: float
    yaw: float       # rotation in horizontal plane
    pitch: float     # rotation around camera X axis
    roll: float      # rotation around camera Y axis
    # Initial GNSS/DLS priors
    gps_easting: float
    gps_northing: float
    gps_altitude: float
    dls_yaw: float
    dls_pitch: float
    dls_roll: float

    def rotation_matrix_world_to_cam(self) -> List[float]:
        """
        Compute 3x3 orthogonal rotation matrix R in SO(3) mapping world vector
        (dE, dN, dZ_up) to camera coordinates (Xc, Yc, Zc_forward_down).

        R = Rx(pitch) * Ry(roll) * R_mount(yaw)
        where R_mount(yaw) = [[-sin(yaw), cos(yaw), 0],
                              [ cos(yaw), sin(yaw), 0],
                              [        0,        0,-1]]
        """
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        # Base mounting matrix M (det = +1)
        m00, m01, m02 = -sy, cy, 0.0
        m10, m11, m12 = cy, sy, 0.0
        m20, m21, m22 = 0.0, 0.0, -1.0

        # Apply Ry(roll)
        cr = math.cos(self.roll)
        sr = math.sin(self.roll)
        ry00 = cr * m00 + sr * m20
        ry01 = cr * m01 + sr * m21
        ry02 = cr * m02 + sr * m22
        ry10, ry11, ry12 = m10, m11, m12
        ry20 = -sr * m00 + cr * m20
        ry21 = -sr * m01 + cr * m21
        ry22 = -sr * m02 + cr * m22

        # Apply Rx(pitch)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        r00, r01, r02 = ry00, ry01, ry02
        r10 = cp * ry10 - sp * ry20
        r11 = cp * ry11 - sp * ry21
        r12 = cp * ry12 - sp * ry22
        r20 = sp * ry10 + cp * ry20
        r21 = sp * ry11 + cp * ry21
        r22 = sp * ry12 + cp * ry22

        return [r00, r01, r02, r10, r11, r12, r20, r21, r22]


@dataclass
class TiePoint3D:
    track_id: int
    easting: float
    northing: float
    elevation: float
    observations: List[Tuple[int, float, float]] = field(default_factory=list)  # (capture_index, undist_u, undist_v)
    is_check_point: bool = False
    mean_reproj_error_px: float = 0.0


@dataclass
class SfMResult:
    poses: Dict[int, CameraPose]
    tie_points: List[TiePoint3D]
    pairwise_matches: List[PairwiseMatch]
    initial_reproj_rmse_px: float
    final_reproj_rmse_px: float
    check_point_rmse_px: float
    mean_gps_XY_shift_m: float
    mean_gps_Z_shift_m: float
    connected_captures: int


def _solve_linear_3x3(A: List[List[float]], b: List[float]) -> Optional[Tuple[float, float, float]]:
    """Solve 3x3 linear system Ax = b via Gaussian elimination with partial pivoting."""
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(3):
        pivot = col
        for r in range(col + 1, 3):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        if abs(M[pivot][col]) < 1e-12:
            return None
        M[col], M[pivot] = M[pivot], M[col]
        div = M[col][col]
        for c in range(col, 4):
            M[col][c] /= div
        for r in range(3):
            if r != col:
                factor = M[r][col]
                for c in range(col, 4):
                    M[r][c] -= factor * M[col][c]
    return M[0][3], M[1][3], M[2][3]


def _solve_linear_nxn(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve NxN linear system Ax = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        if abs(M[pivot][col]) < 1e-12:
            return None
        M[col], M[pivot] = M[pivot], M[col]
        div = M[col][col]
        for c in range(col, n + 1):
            M[col][c] /= div
        for r in range(n):
            if r != col and abs(M[r][col]) > 0.0:
                factor = M[r][col]
                for c in range(col, n + 1):
                    M[r][c] -= factor * M[col][c]
    return [M[i][n] for i in range(n)]


def project_point_undistorted(
    pose: CameraPose,
    R: List[float],
    easting: float,
    northing: float,
    elevation: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Optional[Tuple[float, float]]:
    """Project 3D UTM point (E, N, Z) into undistorted camera pixel coordinates (u, v)."""
    dx = easting - pose.easting
    dy = northing - pose.northing
    dz = elevation - pose.altitude
    xc = R[0] * dx + R[1] * dy + R[2] * dz
    yc = R[3] * dx + R[4] * dy + R[5] * dz
    zc = R[6] * dx + R[7] * dy + R[8] * dz
    if zc <= 0.5:
        return None
    u = fx * (xc / zc) + cx
    v = fy * (yc / zc) + cy
    return u, v


def triangulate_track_3d(
    track_obs: List[Tuple[int, float, float]],
    poses: Dict[int, CameraPose],
    rotations: Dict[int, List[float]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    prior_ground_z: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Triangulate a 3D point from multiple camera rays by minimizing perpendicular ray distances,
    regularized with a gentle ground-plane elevation prior for narrow-baseline stability.
    """
    A = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    rhs = [0.0, 0.0, 0.0]

    for cap_idx, u, v in track_obs:
        pose = poses[cap_idx]
        R = rotations[cap_idx]
        xn = (u - cx) / fx
        yn = (v - cy) / fy
        # Ray in camera coords is (xn, yn, 1.0); transform to world via R^T
        wx = R[0] * xn + R[3] * yn + R[6] * 1.0
        wy = R[1] * xn + R[4] * yn + R[7] * 1.0
        wz = R[2] * xn + R[5] * yn + R[8] * 1.0
        norm = math.sqrt(wx * wx + wy * wy + wz * wz)
        if norm < 1e-12:
            continue
        dx, dy, dz = wx / norm, wy / norm, wz / norm

        # Projection matrix P_perp = I - d * d^T
        p00 = 1.0 - dx * dx
        p01 = -dx * dy
        p02 = -dx * dz
        p11 = 1.0 - dy * dy
        p12 = -dy * dz
        p22 = 1.0 - dz * dz

        A[0][0] += p00; A[0][1] += p01; A[0][2] += p02
        A[1][0] += p01; A[1][1] += p11; A[1][2] += p12
        A[2][0] += p02; A[2][1] += p12; A[2][2] += p22

        cx_w, cy_w, cz_w = pose.easting, pose.northing, pose.altitude
        rhs[0] += p00 * cx_w + p01 * cy_w + p02 * cz_w
        rhs[1] += p01 * cx_w + p11 * cy_w + p12 * cz_w
        rhs[2] += p02 * cx_w + p12 * cy_w + p22 * cz_w

    # Weak elevation regularization (sigma_z_ground ~ 0.65m on flat turfgrass research plots)
    w_z_prior = 0.08
    A[2][2] += w_z_prior
    rhs[2] += w_z_prior * prior_ground_z

    sol = _solve_linear_3x3(A, rhs)
    if sol is None:
        return None
    e, n, z = sol
    if abs(z - prior_ground_z) > 6.0:
        z = max(prior_ground_z - 3.0, min(prior_ground_z + 3.0, z))
    return e, n, z


def run_sfm_and_bundle_adjustment(
    survey_captures: List[CaptureEvent],
    ground_elevation_m: float,
    rad_model: Optional[RadiometricCalibrationModel] = None,
    ref_band_index: int = 2,
    ba_iterations: int = 12,
) -> SfMResult:
    """
    Execute Stage 2 & Stage 3:
    1. Extract multiscale Shi-Tomasi keypoints & 64-D gradient descriptors across all survey captures.
    2. Build candidate spatial overlap graph (along-track + cross-track neighbor pairs) and run
       mutual ratio-test matching + 2D similarity RANSAC.
    3. Estimate global sensor yaw alignment and optimize the 2D/3D pose graph over all cameras.
    4. Build multi-view feature tracks and run iterative Levenberg-Marquardt Bundle Adjustment
       jointly refining camera exterior orientations (E, N, Z, yaw, pitch, roll) and 3D tie points.
    """
    # 1. Extract features for all survey captures in parallel
    def _extract(cap: CaptureEvent) -> FrameFeatures:
        return extract_features_for_capture(
            cap,
            band_index=ref_band_index,
            rad_model=rad_model,
            downsample_factor=2,
            max_features=750,
        )

    with ThreadPoolExecutor(max_workers=12) as ex:
        features_list = list(ex.map(_extract, survey_captures))

    feat_by_idx: Dict[int, FrameFeatures] = {f.capture_index: f for f in features_list}
    cap_by_idx: Dict[int, CaptureEvent] = {c.capture_index: c for c in survey_captures}

    ref_meta = survey_captures[0].bands[ref_band_index]
    fx = ref_meta.fx_px
    fy = ref_meta.fy_px
    cx = ref_meta.cx_px
    cy = ref_meta.cy_px
    mean_agl = sum(c.altitude_m for c in survey_captures) / len(survey_captures) - ground_elevation_m
    mean_gsd = mean_agl / fx

    # 2. Construct candidate image pairs (along-track i+1, i+2 and cross-track spatial neighbors < 11.5m)
    candidate_pairs: List[Tuple[int, int, str, float, float]] = []
    seen_pairs = set()
    n_caps = len(survey_captures)

    for i in range(n_caps):
        c1 = survey_captures[i]
        # Along-track neighbors
        for step in (1, 2):
            if i + step < n_caps:
                c2 = survey_captures[i + step]
                dist = math.hypot(c2.easting_m - c1.easting_m, c2.northing_m - c1.northing_m)
                if dist < 12.0:
                    key = (min(c1.capture_index, c2.capture_index), max(c1.capture_index, c2.capture_index))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        # Expected pixel shift from GPS delta: du ~ -dN / gsd, dv ~ -dE / gsd
                        du0 = -(c2.northing_m - c1.northing_m) / mean_gsd
                        dv0 = -(c2.easting_m - c1.easting_m) / mean_gsd
                        candidate_pairs.append((c1.capture_index, c2.capture_index, "along_track", du0, dv0))

        # Cross-track neighbors in adjacent flight lines
        cross_candidates: List[Tuple[float, CaptureEvent]] = []
        for j in range(i + 3, n_caps):
            c2 = survey_captures[j]
            if abs(c2.capture_index - c1.capture_index) <= 3:
                continue
            dist = math.hypot(c2.easting_m - c1.easting_m, c2.northing_m - c1.northing_m)
            if dist < 8.5:
                cross_candidates.append((dist, c2))
        cross_candidates.sort(key=lambda x: x[0])
        for _, c2 in cross_candidates[:3]:
            key = (min(c1.capture_index, c2.capture_index), max(c1.capture_index, c2.capture_index))
            if key not in seen_pairs:
                seen_pairs.add(key)
                du0 = -(c2.northing_m - c1.northing_m) / mean_gsd
                dv0 = -(c2.easting_m - c1.easting_m) / mean_gsd
                candidate_pairs.append((c1.capture_index, c2.capture_index, "cross_track", du0, dv0))

    # Match all candidate pairs in parallel
    def _match_pair(item: Tuple[int, int, str, float, float]) -> Optional[PairwiseMatch]:
        idx1, idx2, ptype, du0, dv0 = item
        return match_feature_pair(
            feat_by_idx[idx1],
            feat_by_idx[idx2],
            pair_type=ptype,
            prior_du=du0,
            prior_dv=dv0,
            search_radius_px=950.0,
            ratio_thresh=0.88,
            ransac_thresh_px=6.0,
            ransac_iters=400,
        )

    with ThreadPoolExecutor(max_workers=12) as ex:
        raw_matches = list(ex.map(_match_pair, candidate_pairs))

    valid_matches = [m for m in raw_matches if m is not None and m.num_inliers >= 8]

    # 3. Estimate global yaw mounting offset from high-confidence along-track pairs
    yaw_samples: List[float] = []
    for m in valid_matches:
        if m.pair_type == "along_track" and m.num_inliers >= 25:
            c1 = cap_by_idx[m.idx1]
            c2 = cap_by_idx[m.idx2]
            de = c2.easting_m - c1.easting_m
            dn = c2.northing_m - c1.northing_m
            if math.hypot(de, dn) > 1.2:
                _, _, tx, ty = m.similarity_params
                vx = -tx * mean_gsd
                vy = -ty * mean_gsd
                cos_y = vy * de + vx * dn
                sin_y = vy * dn - vx * de
                est_yaw = math.atan2(sin_y, cos_y)
                yaw_samples.append(est_yaw)

    yaw_samples.sort()
    median_yaw = yaw_samples[len(yaw_samples) // 2] if yaw_samples else 0.0

    # Initialize camera poses
    poses: Dict[int, CameraPose] = {}
    for c in survey_captures:
        poses[c.capture_index] = CameraPose(
            capture_index=c.capture_index,
            easting=c.easting_m,
            northing=c.northing_m,
            altitude=c.altitude_m,
            yaw=median_yaw,
            pitch=0.0,
            roll=0.0,
            gps_easting=c.easting_m,
            gps_northing=c.northing_m,
            gps_altitude=c.altitude_m,
            dls_yaw=c.dls_yaw_rad,
            dls_pitch=c.dls_pitch_rad,
            dls_roll=c.dls_roll_rad,
        )

    # 4. Pose-Graph Gauss-Seidel / SOR Relaxation over camera centers (E_i, N_i, Z_i) and yaw_i
    w_gps_xy = 0.25  # smooths ~1-2m GPS jitter while preserving global georeferencing
    w_gps_z = 0.50
    for _iter in range(40):
        new_e: Dict[int, float] = {}
        new_n: Dict[int, float] = {}
        new_z: Dict[int, float] = {}
        new_yaw: Dict[int, float] = {}

        adj: Dict[int, List[Tuple[int, float, float, float, float, float]]] = {c.capture_index: [] for c in survey_captures}
        for m in valid_matches:
            a, b, tx, ty = m.similarity_params
            scale = math.hypot(a, b)
            d_yaw = math.atan2(b, a)
            p1 = poses[m.idx1]
            p2 = poses[m.idx2]
            ym = 0.5 * (p1.yaw + p2.yaw)
            cy_m = math.cos(ym)
            sy_m = math.sin(ym)
            agl_m = 0.5 * (p1.altitude + p2.altitude) - ground_elevation_m
            gsd_m = agl_m / fx

            dx_c = -tx * gsd_m
            dy_c = -ty * gsd_m
            de_rel = -sy_m * dx_c + cy_m * dy_c
            dn_rel = cy_m * dx_c + sy_m * dy_c
            dz_rel = (1.0 - scale) * agl_m * 0.25
            weight = min(3.0, math.sqrt(m.num_inliers / 25.0))

            adj[m.idx1].append((m.idx2, -de_rel, -dn_rel, -dz_rel, -d_yaw, weight))
            adj[m.idx2].append((m.idx1, de_rel, dn_rel, dz_rel, d_yaw, weight))

        for c in survey_captures:
            idx = c.capture_index
            p = poses[idx]
            sum_we = w_gps_xy * p.gps_easting
            sum_wn = w_gps_xy * p.gps_northing
            sum_wz = w_gps_z * p.gps_altitude
            sum_wy = 0.5 * median_yaw
            w_tot_xy = w_gps_xy
            w_tot_z = w_gps_z
            w_tot_y = 0.5

            for nbr_idx, de_rel, dn_rel, dz_rel, dyaw_rel, w in adj[idx]:
                pn = poses[nbr_idx]
                sum_we += w * (pn.easting + de_rel)
                sum_wn += w * (pn.northing + dn_rel)
                sum_wz += (0.30 * w) * (pn.altitude + dz_rel)
                sum_wy += w * (pn.yaw + dyaw_rel)
                w_tot_xy += w
                w_tot_z += 0.30 * w
                w_tot_y += w

            new_e[idx] = sum_we / w_tot_xy
            new_n[idx] = sum_wn / w_tot_xy
            new_z[idx] = sum_wz / w_tot_z
            new_yaw[idx] = sum_wy / w_tot_y

        for idx in poses:
            poses[idx].easting = new_e[idx]
            poses[idx].northing = new_n[idx]
            poses[idx].altitude = new_z[idx]
            poses[idx].yaw = new_yaw[idx]

    # 5. Build Multi-View Feature Tracks using Union-Find over (capture_index, kp_idx)
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(node: Tuple[int, int]) -> Tuple[int, int]:
        path = []
        cur = node
        while cur in parent and parent[cur] != cur:
            path.append(cur)
            cur = parent[cur]
        if cur not in parent:
            parent[cur] = cur
        for nd in path:
            parent[nd] = cur
        return cur

    def union(n1: Tuple[int, int], n2: Tuple[int, int]) -> None:
        r1, r2 = find(n1), find(n2)
        if r1 != r2:
            parent[r2] = r1

    for m in valid_matches:
        for kp1, kp2 in m.inlier_kp_pairs:
            union((m.idx1, kp1), (m.idx2, kp2))

    raw_tracks: Dict[Tuple[int, int], Dict[int, Tuple[float, float]]] = {}
    for node in list(parent.keys()):
        root = find(node)
        cap_idx, kp_idx = node
        f = feat_by_idx[cap_idx]
        u = f.keypoints_undist_uv[2 * kp_idx]
        v = f.keypoints_undist_uv[2 * kp_idx + 1]
        if root not in raw_tracks:
            raw_tracks[root] = {}
        if cap_idx not in raw_tracks[root]:
            raw_tracks[root][cap_idx] = (u, v)

    # Filter tracks with >= 2 views and cap total tracks for fast, high-precision Bundle Adjustment
    sorted_tracks = sorted(raw_tracks.values(), key=lambda d: len(d), reverse=True)
    tie_points: List[TiePoint3D] = []
    rotations = {idx: p.rotation_matrix_world_to_cam() for idx, p in poses.items()}

    for t_idx, obs_dict in enumerate(sorted_tracks):
        if len(obs_dict) < 2:
            continue
        obs_list = [(c_idx, uv[0], uv[1]) for c_idx, uv in sorted(obs_dict.items())]
        tri = triangulate_track_3d(obs_list, poses, rotations, fx, fy, cx, cy, ground_elevation_m)
        if tri is None:
            continue
        errs = []
        for c_idx, u, v in obs_list:
            proj = project_point_undistorted(poses[c_idx], rotations[c_idx], tri[0], tri[1], tri[2], fx, fy, cx, cy)
            if proj is not None:
                errs.append(math.hypot(proj[0] - u, proj[1] - v))
        if len(errs) == len(obs_list) and (sum(errs) / len(errs)) < 25.0:
            is_check = (len(tie_points) % 10 == 9) and (len(obs_list) >= 3)
            tie_points.append(
                TiePoint3D(
                    track_id=len(tie_points),
                    easting=tri[0],
                    northing=tri[1],
                    elevation=tri[2],
                    observations=obs_list,
                    is_check_point=is_check,
                    mean_reproj_error_px=sum(errs) / len(errs),
                )
            )
            if len(tie_points) >= 6000:
                break

    def _compute_rmse(only_check: bool = False) -> float:
        sq_sum = 0.0
        cnt = 0
        for tp in tie_points:
            if tp.is_check_point != only_check:
                continue
            for c_idx, u, v in tp.observations:
                proj = project_point_undistorted(
                    poses[c_idx], rotations[c_idx], tp.easting, tp.northing, tp.elevation, fx, fy, cx, cy
                )
                if proj is not None:
                    du = proj[0] - u
                    dv = proj[1] - v
                    sq_sum += du * du + dv * dv
                    cnt += 1
        return math.sqrt(sq_sum / max(1, cnt))

    initial_rmse = _compute_rmse(only_check=False)

    # 6. Alternating Sparse Levenberg-Marquardt Bundle Adjustment
    obs_by_cam: Dict[int, List[Tuple[int, float, float]]] = {idx: [] for idx in poses}
    for tp_idx, tp in enumerate(tie_points):
        if not tp.is_check_point:
            for c_idx, u, v in tp.observations:
                obs_by_cam[c_idx].append((tp_idx, u, v))

    damping = 2.0
    huber_delta = 4.0  # pixels

    for _ba_iter in range(ba_iterations):
        # Step A: Update camera poses (E, N, Z, yaw)
        for c_idx, cam_obs in obs_by_cam.items():
            if len(cam_obs) < 4:
                continue
            p = poses[c_idx]
            R = rotations[c_idx]
            H = [[0.0] * 4 for _ in range(4)]
            g_vec = [0.0] * 4

            w_prior_xy = 0.15
            w_prior_z = 0.80
            w_prior_yaw = 50.0
            H[0][0] += w_prior_xy + damping
            H[1][1] += w_prior_xy + damping
            H[2][2] += w_prior_z + damping
            H[3][3] += w_prior_yaw + damping * 10.0

            g_vec[0] -= w_prior_xy * (p.easting - p.gps_easting)
            g_vec[1] -= w_prior_xy * (p.northing - p.gps_northing)
            g_vec[2] -= w_prior_z * (p.altitude - p.gps_altitude)
            g_vec[3] -= w_prior_yaw * (p.yaw - median_yaw)

            for tp_idx, u_obs, v_obs in cam_obs:
                tp = tie_points[tp_idx]
                dx = tp.easting - p.easting
                dy = tp.northing - p.northing
                dz = tp.elevation - p.altitude

                xc = R[0] * dx + R[1] * dy + R[2] * dz
                yc = R[3] * dx + R[4] * dy + R[5] * dz
                zc = R[6] * dx + R[7] * dy + R[8] * dz
                if zc <= 1.0:
                    continue
                inv_z = 1.0 / zc
                inv_z2 = inv_z * inv_z

                u_pred = fx * xc * inv_z + cx
                v_pred = fy * yc * inv_z + cy
                ru = u_pred - u_obs
                rv = v_pred - v_obs
                err = math.hypot(ru, rv)
                if err > 25.0:
                    continue
                w_huber = 1.0 if err <= huber_delta else (huber_delta / err)

                du_dxc = fx * inv_z
                du_dzc = -fx * xc * inv_z2
                dv_dyc = fy * inv_z
                dv_dzc = -fy * yc * inv_z2

                ju = [
                    -(du_dxc * R[0] + du_dzc * R[6]),
                    -(du_dxc * R[1] + du_dzc * R[7]),
                    -(du_dxc * R[2] + du_dzc * R[8]),
                    du_dxc * (-yc),
                ]
                jv = [
                    -(dv_dyc * R[3] + dv_dzc * R[6]),
                    -(dv_dyc * R[4] + dv_dzc * R[7]),
                    -(dv_dyc * R[5] + dv_dzc * R[8]),
                    dv_dyc * xc,
                ]

                for r_i in range(4):
                    g_vec[r_i] -= w_huber * (ju[r_i] * ru + jv[r_i] * rv)
                    for c_i in range(4):
                        H[r_i][c_i] += w_huber * (ju[r_i] * ju[c_i] + jv[r_i] * jv[c_i])

            step = _solve_linear_nxn(H, g_vec)
            if step is not None:
                p.easting += max(-0.5, min(0.5, step[0]))
                p.northing += max(-0.5, min(0.5, step[1]))
                p.altitude += max(-0.20, min(0.20, step[2]))
                p.yaw += max(-0.02, min(0.02, step[3]))
                rotations[c_idx] = p.rotation_matrix_world_to_cam()

        # Step B: Re-triangulate 3D tie points from updated camera poses
        for tp in tie_points:
            tri = triangulate_track_3d(tp.observations, poses, rotations, fx, fy, cx, cy, ground_elevation_m)
            if tri is not None:
                tp.easting, tp.northing, tp.elevation = tri

    # Update final per-point reprojection errors
    for tp in tie_points:
        errs = []
        for c_idx, u, v in tp.observations:
            proj = project_point_undistorted(
                poses[c_idx], rotations[c_idx], tp.easting, tp.northing, tp.elevation, fx, fy, cx, cy
            )
            if proj is not None:
                errs.append(math.hypot(proj[0] - u, proj[1] - v))
        tp.mean_reproj_error_px = sum(errs) / len(errs) if errs else 0.0

    final_rmse = _compute_rmse(only_check=False)
    check_rmse = _compute_rmse(only_check=True)

    xy_shifts = [math.hypot(p.easting - p.gps_easting, p.northing - p.gps_northing) for p in poses.values()]
    z_shifts = [abs(p.altitude - p.gps_altitude) for p in poses.values()]

    connected_set = set()
    for m in valid_matches:
        connected_set.add(m.idx1)
        connected_set.add(m.idx2)

    return SfMResult(
        poses=poses,
        tie_points=tie_points,
        pairwise_matches=valid_matches,
        initial_reproj_rmse_px=initial_rmse,
        final_reproj_rmse_px=final_rmse,
        check_point_rmse_px=check_rmse,
        mean_gps_XY_shift_m=sum(xy_shifts) / max(1, len(xy_shifts)),
        mean_gps_Z_shift_m=sum(z_shifts) / max(1, len(z_shifts)),
        connected_captures=len(connected_set),
    )
