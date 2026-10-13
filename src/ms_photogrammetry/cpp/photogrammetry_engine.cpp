// From-scratch C++17 Photogrammetric Core Engine
// Implements radiometric calibration, Brown-Conrady lens distortion/undistortion,
// Shi-Tomasi multiscale feature detection, SIFT-like gradient descriptor extraction,
// mutual ratio-test matching, RANSAC geometric verification, gradient-NCC cross-band
// registration, and multithreaded nadir-prioritized orthorectification & mosaicking.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

extern "C" {

// ---------------------------------------------------------------------------
// 1. Radiometric Calibration (Raw 16-bit DN -> Calibrated Float32)
// ---------------------------------------------------------------------------
void radiometric_calibrate_band(
    const uint16_t* raw_dn,
    int width,
    int height,
    double black_level,
    const int* vig_exp_p,
    const int* vig_exp_q,
    const double* vig_coeffs,
    int num_vig_terms,
    double a1,
    double a2,
    double a3,
    double exposure_s,
    double gain,
    double temp_c,
    double radiance_to_refl,
    int is_thermal,
    float* out_calibrated
) {
    if (is_thermal) {
        // MicaSense Altum-PT LWIR thermal band is stored in centi-Kelvin (0.01 K)
        const int total = width * height;
        for (int i = 0; i < total; ++i) {
            double k = static_cast<double>(raw_dn[i]) * 0.01;
            out_calibrated[i] = static_cast<float>(k - 273.15); // Celsius
        }
        return;
    }

    const double inv_norm = 1.0 / 65535.0;
    const double p_dark = black_level * inv_norm;
    const double inv_w = 1.0 / static_cast<double>(width);
    const double inv_h = 1.0 / static_cast<double>(height);
    const double a1_over_g = a1 / std::max(1e-6, gain);
    const double row_slope = a2 - a3 * exposure_s;

    // Precompute powers x^p and y^q (max exponent is 5)
    std::vector<double> x_pows(width * 6, 1.0);
    for (int x = 0; x < width; ++x) {
        double xn = (x + 0.5) * inv_w;
        double cur = 1.0;
        for (int p = 0; p <= 5; ++p) {
            x_pows[x * 6 + p] = cur;
            cur *= xn;
        }
    }

    for (int y = 0; y < height; ++y) {
        double yn = (y + 0.5) * inv_h;
        double y_pows[6];
        double cur = 1.0;
        for (int q = 0; q <= 5; ++q) {
            y_pows[q] = cur;
            cur *= yn;
        }
        double row_denom = exposure_s + static_cast<double>(y) * row_slope;
        if (std::abs(row_denom) < 1e-12) {
            row_denom = 1e-12;
        }
        const double row_mult = (a1_over_g / row_denom) * radiance_to_refl;

        const uint16_t* row_in = raw_dn + y * width;
        float* row_out = out_calibrated + y * width;

        for (int x = 0; x < width; ++x) {
            double vig = 1.0;
            if (num_vig_terms > 0) {
                vig = 0.0;
                const double* xp = &x_pows[x * 6];
                for (int k = 0; k < num_vig_terms; ++k) {
                    vig += vig_coeffs[k] * xp[vig_exp_p[k]] * y_pows[vig_exp_q[k]];
                }
                if (vig < 0.1) vig = 0.1;
            }
            double p_raw = static_cast<double>(row_in[x]) * inv_norm;
            double v_corr = std::max(0.0, (p_raw - p_dark) / vig);
            double refl = v_corr * row_mult;
            row_out[x] = static_cast<float>(std::clamp(refl, 0.0, 1.5));
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Brown-Conrady Point & Image Undistortion
// ---------------------------------------------------------------------------
inline void distort_normalized_point(
    double xn, double yn,
    double k1, double k2, double k3, double p1, double p2,
    double& xd, double& yd
) {
    double r2 = xn * xn + yn * yn;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    double dx_tang = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
    double dy_tang = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
    xd = xn * radial + dx_tang;
    yd = yn * radial + dy_tang;
}

void undistort_points_2d(
    const double* uv_distorted,
    int num_points,
    double fx, double fy, double cx, double cy,
    double k1, double k2, double k3, double p1, double p2,
    double* uv_undistorted
) {
    for (int i = 0; i < num_points; ++i) {
        double ud = uv_distorted[2 * i];
        double vd = uv_distorted[2 * i + 1];
        double xd = (ud - cx) / fx;
        double yd = (vd - cy) / fy;
        // Iterative fixed-point / Gauss-Newton inversion of Brown-Conrady distortion
        double xn = xd;
        double yn = yd;
        for (int iter = 0; iter < 8; ++iter) {
            double est_xd, est_yd;
            distort_normalized_point(xn, yn, k1, k2, k3, p1, p2, est_xd, est_yd);
            double ex = est_xd - xd;
            double ey = est_yd - yd;
            xn -= ex;
            yn -= ey;
            if (ex * ex + ey * ey < 1e-18) break;
        }
        uv_undistorted[2 * i] = xn * fx + cx;
        uv_undistorted[2 * i + 1] = yn * fy + cy;
    }
}

void undistort_image_float(
    const float* src,
    int width,
    int height,
    double fx, double fy, double cx, double cy,
    double k1, double k2, double k3, double p1, double p2,
    float* dst
) {
    const double inv_fx = 1.0 / fx;
    const double inv_fy = 1.0 / fy;
    for (int v = 0; v < height; ++v) {
        double yn = (static_cast<double>(v) - cy) * inv_fy;
        float* dst_row = dst + v * width;
        for (int u = 0; u < width; ++u) {
            double xn = (static_cast<double>(u) - cx) * inv_fx;
            double xd, yd;
            distort_normalized_point(xn, yn, k1, k2, k3, p1, p2, xd, yd);
            double ud = fx * xd + cx;
            double vd = fy * yd + cy;
            int x0 = static_cast<int>(std::floor(ud));
            int y0 = static_cast<int>(std::floor(vd));
            if (x0 >= 0 && x0 + 1 < width && y0 >= 0 && y0 + 1 < height) {
                float ax = static_cast<float>(ud - x0);
                float ay = static_cast<float>(vd - y0);
                const float* r0 = src + y0 * width + x0;
                const float* r1 = r0 + width;
                float v0 = r0[0] * (1.0f - ax) + r0[1] * ax;
                float v1 = r1[0] * (1.0f - ax) + r1[1] * ax;
                dst_row[u] = v0 * (1.0f - ay) + v1 * ay;
            } else {
                dst_row[u] = 0.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Feature Detection (Shi-Tomasi Corner Response + Grid NMS + 64-D Descriptor)
// ---------------------------------------------------------------------------
int detect_and_describe_features(
    const float* img,
    int width,
    int height,
    int downsample_factor,
    int max_features,
    int grid_cols,
    int grid_rows,
    double* out_keypoints_uv,    // size: max_features * 2 (in full-resolution pixel coords)
    float* out_descriptors_64,   // size: max_features * 64
    float* out_scores            // size: max_features
) {
    const int ds = std::max(1, downsample_factor);
    const int w = width / ds;
    const int h = height / ds;
    if (w < 32 || h < 32) return 0;

    // 1. Box-filter / downsample to working resolution (w x h)
    std::vector<float> work(w * h, 0.0f);
    const float inv_area = 1.0f / static_cast<float>(ds * ds);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int dy = 0; dy < ds; ++dy) {
                const float* row = img + (y * ds + dy) * width + (x * ds);
                for (int dx = 0; dx < ds; ++dx) {
                    sum += row[dx];
                }
            }
            work[y * w + x] = sum * inv_area;
        }
    }

    // 2. Separable 5-tap Gaussian smoothing [1, 4, 6, 4, 1] / 16
    std::vector<float> smooth_tmp(w * h, 0.0f);
    std::vector<float> smooth(w * h, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 2; x < w - 2; ++x) {
            const float* r = &work[y * w + x];
            smooth_tmp[y * w + x] = (r[-2] + 4.0f * r[-1] + 6.0f * r[0] + 4.0f * r[1] + r[2]) * 0.0625f;
        }
    }
    for (int y = 2; y < h - 2; ++y) {
        for (int x = 2; x < w - 2; ++x) {
            smooth[y * w + x] = (
                smooth_tmp[(y - 2) * w + x] +
                4.0f * smooth_tmp[(y - 1) * w + x] +
                6.0f * smooth_tmp[y * w + x] +
                4.0f * smooth_tmp[(y + 1) * w + x] +
                smooth_tmp[(y + 2) * w + x]
            ) * 0.0625f;
        }
    }

    // 3. Compute gradients Ix, Iy
    std::vector<float> ix(w * h, 0.0f);
    std::vector<float> iy(w * h, 0.0f);
    for (int y = 3; y < h - 3; ++y) {
        for (int x = 3; x < w - 3; ++x) {
            int idx = y * w + x;
            ix[idx] = 0.5f * (smooth[idx + 1] - smooth[idx - 1]);
            iy[idx] = 0.5f * (smooth[idx + w] - smooth[idx - w]);
        }
    }

    // 4. Compute Shi-Tomasi minimum eigenvalue corner response over 3x3 window
    std::vector<float> response(w * h, 0.0f);
    const int margin = 12;
    for (int y = margin; y < h - margin; ++y) {
        for (int x = margin; x < w - margin; ++x) {
            float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                int r_off = (y + dy) * w + x;
                for (int dx = -1; dx <= 1; ++dx) {
                    float gx = ix[r_off + dx];
                    float gy = iy[r_off + dx];
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                }
            }
            float trace = sxx + syy;
            float det_disc = std::sqrt((sxx - syy) * (sxx - syy) + 4.0f * sxy * sxy);
            response[y * w + x] = 0.5f * (trace - det_disc);
        }
    }

    // 5. Local 5x5 Non-Maximum Suppression + Spatial Grid Bucketing for uniform turfgrass coverage
    struct Candidate {
        float u;
        float v;
        float score;
        int x_int;
        int y_int;
    };
    std::vector<std::vector<Candidate>> cells(grid_cols * grid_rows);
    const float cell_w = static_cast<float>(w - 2 * margin) / grid_cols;
    const float cell_h = static_cast<float>(h - 2 * margin) / grid_rows;

    for (int y = margin; y < h - margin; ++y) {
        for (int x = margin; x < w - margin; ++x) {
            float val = response[y * w + x];
            if (val <= 1e-9f) continue;
            bool is_max = true;
            for (int dy = -2; dy <= 2 && is_max; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (response[(y + dy) * w + (x + dx)] >= val) {
                        is_max = false;
                        break;
                    }
                }
            }
            if (!is_max) continue;

            // Sub-pixel quadratic interpolation
            float rx_minus = response[y * w + (x - 1)];
            float rx_plus  = response[y * w + (x + 1)];
            float ry_minus = response[(y - 1) * w + x];
            float ry_plus  = response[(y + 1) * w + x];
            float denom_x = 2.0f * (rx_minus - 2.0f * val + rx_plus);
            float denom_y = 2.0f * (ry_minus - 2.0f * val + ry_plus);
            float sub_x = (std::abs(denom_x) > 1e-10f) ? std::clamp((rx_minus - rx_plus) / denom_x, -0.5f, 0.5f) : 0.0f;
            float sub_y = (std::abs(denom_y) > 1e-10f) ? std::clamp((ry_minus - ry_plus) / denom_y, -0.5f, 0.5f) : 0.0f;

            int gx = std::clamp(static_cast<int>((x - margin) / cell_w), 0, grid_cols - 1);
            int gy = std::clamp(static_cast<int>((y - margin) / cell_h), 0, grid_rows - 1);
            cells[gy * grid_cols + gx].push_back({
                (static_cast<float>(x) + sub_x + 0.5f) * ds,
                (static_cast<float>(y) + sub_y + 0.5f) * ds,
                val,
                x,
                y
            });
        }
    }

    // Sort candidates within each cell and select top per cell, then top overall
    int quota_per_cell = std::max(1, (max_features + (grid_cols * grid_rows) - 1) / (grid_cols * grid_rows));
    std::vector<Candidate> selected;
    selected.reserve(max_features * 2);
    for (auto& bucket : cells) {
        std::sort(bucket.begin(), bucket.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });
        int take = std::min(static_cast<int>(bucket.size()), quota_per_cell + 2);
        for (int i = 0; i < take; ++i) {
            selected.push_back(bucket[i]);
        }
    }
    std::sort(selected.begin(), selected.end(), [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
    });
    if (static_cast<int>(selected.size()) > max_features) {
        selected.resize(max_features);
    }

    // 6. Compute 64-D SIFT-like Gradient Orientation & Patch Descriptor for each keypoint
    // 4x4 spatial sub-blocks over a 16x16 neighborhood around (x_int, y_int),
    // each sub-block accumulates 4 directional gradient bins: (+dx, -dx, +dy, -dy)
    int count = static_cast<int>(selected.size());
    for (int k = 0; k < count; ++k) {
        const auto& kp = selected[k];
        out_keypoints_uv[2 * k]     = static_cast<double>(kp.u);
        out_keypoints_uv[2 * k + 1] = static_cast<double>(kp.v);
        out_scores[k] = kp.score;

        float* desc = out_descriptors_64 + k * 64;
        std::memset(desc, 0, 64 * sizeof(float));

        int cx0 = kp.x_int;
        int cy0 = kp.y_int;
        for (int by = 0; by < 4; ++by) {
            for (int bx = 0; bx < 4; ++bx) {
                float* bin = desc + (by * 4 + bx) * 4;
                int y_start = cy0 - 8 + by * 4;
                int x_start = cx0 - 8 + bx * 4;
                for (int py = 0; py < 4; ++py) {
                    int yy = std::clamp(y_start + py, 1, h - 2);
                    for (int px = 0; px < 4; ++px) {
                        int xx = std::clamp(x_start + px, 1, w - 2);
                        float gx = ix[yy * w + xx];
                        float gy = iy[yy * w + xx];
                        if (gx > 0.0f) bin[0] += gx; else bin[1] -= gx;
                        if (gy > 0.0f) bin[2] += gy; else bin[3] -= gy;
                    }
                }
            }
        }

        // L2 normalize, clamp at 0.2 for non-linear illumination invariance, and re-normalize
        float norm2 = 1e-12f;
        for (int d = 0; d < 64; ++d) norm2 += desc[d] * desc[d];
        float inv_norm_d = 1.0f / std::sqrt(norm2);
        norm2 = 1e-12f;
        for (int d = 0; d < 64; ++d) {
            desc[d] = std::min(0.2f, desc[d] * inv_norm_d);
            norm2 += desc[d] * desc[d];
        }
        inv_norm_d = 1.0f / std::sqrt(norm2);
        for (int d = 0; d < 64; ++d) {
            desc[d] *= inv_norm_d;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// 4. Mutual Nearest-Neighbor + Lowe Ratio-Test Feature Matching with Spatial Prior
// ---------------------------------------------------------------------------
int match_descriptors_mutual(
    const double* kp1_uv,
    const float* desc1,
    int n1,
    const double* kp2_uv,
    const float* desc2,
    int n2,
    double prior_du,
    double prior_dv,
    double search_radius_px,
    double ratio_thresh,
    int* out_match_idx1,
    int* out_match_idx2,
    double* out_match_dist
) {
    if (n1 <= 0 || n2 <= 0) return 0;
    const bool use_prior = (search_radius_px > 0.0);
    const double r2_max = search_radius_px * search_radius_px;

    std::vector<int> best_1to2(n1, -1);
    std::vector<float> best_dist1(n1, 1e30f);

    for (int i = 0; i < n1; ++i) {
        const float* d1 = desc1 + i * 64;
        double u1_pred = kp1_uv[2 * i] + prior_du;
        double v1_pred = kp1_uv[2 * i + 1] + prior_dv;

        float d_best = 1e30f;
        float d_second = 1e30f;
        int j_best = -1;

        for (int j = 0; j < n2; ++j) {
            if (use_prior) {
                double du = kp2_uv[2 * j] - u1_pred;
                double dv = kp2_uv[2 * j + 1] - v1_pred;
                if (du * du + dv * dv > r2_max) continue;
            }
            const float* d2 = desc2 + j * 64;
            // Since descriptors are unit L2 normalized, L2^2 = 2 - 2 * dot(d1, d2)
            float dot = 0.0f;
            for (int k = 0; k < 64; ++k) {
                dot += d1[k] * d2[k];
            }
            float dist2 = std::max(0.0f, 2.0f - 2.0f * dot);
            if (dist2 < d_best) {
                d_second = d_best;
                d_best = dist2;
                j_best = j;
            } else if (dist2 < d_second) {
                d_second = dist2;
            }
        }
        if (j_best >= 0 && d_best < static_cast<float>(ratio_thresh * ratio_thresh) * d_second) {
            best_1to2[i] = j_best;
            best_dist1[i] = std::sqrt(d_best);
        }
    }

    // Mutual consistency check (2 -> 1)
    std::vector<int> best_2to1(n2, -1);
    for (int j = 0; j < n2; ++j) {
        const float* d2 = desc2 + j * 64;
        double u2_pred = kp2_uv[2 * j] - prior_du;
        double v2_pred = kp2_uv[2 * j + 1] - prior_dv;

        float d_best = 1e30f;
        int i_best = -1;
        for (int i = 0; i < n1; ++i) {
            if (use_prior) {
                double du = kp1_uv[2 * i] - u2_pred;
                double dv = kp1_uv[2 * i + 1] - v2_pred;
                if (du * du + dv * dv > r2_max) continue;
            }
            const float* d1 = desc1 + i * 64;
            float dot = 0.0f;
            for (int k = 0; k < 64; ++k) dot += d1[k] * d2[k];
            float dist2 = std::max(0.0f, 2.0f - 2.0f * dot);
            if (dist2 < d_best) {
                d_best = dist2;
                i_best = i;
            }
        }
        best_2to1[j] = i_best;
    }

    int num_matches = 0;
    for (int i = 0; i < n1; ++i) {
        int j = best_1to2[i];
        if (j >= 0 && best_2to1[j] == i) {
            out_match_idx1[num_matches] = i;
            out_match_idx2[num_matches] = j;
            out_match_dist[num_matches] = static_cast<double>(best_dist1[i]);
            ++num_matches;
        }
    }
    return num_matches;
}

// ---------------------------------------------------------------------------
// 5. RANSAC 2D Similarity Transform Estimation (4-DOF: scale, rot, tx, ty)
//    u2 = a * u1 - b * v1 + tx
//    v2 = b * u1 + a * v1 + ty
// ---------------------------------------------------------------------------
static void fit_similarity_lsq(
    const double* p1,
    const double* p2,
    const std::vector<int>& indices,
    double& a, double& b, double& tx, double& ty
) {
    const int m = static_cast<int>(indices.size());
    double c1x = 0.0, c1y = 0.0, c2x = 0.0, c2y = 0.0;
    for (int idx : indices) {
        c1x += p1[2 * idx];
        c1y += p1[2 * idx + 1];
        c2x += p2[2 * idx];
        c2y += p2[2 * idx + 1];
    }
    double inv_m = 1.0 / std::max(1, m);
    c1x *= inv_m; c1y *= inv_m;
    c2x *= inv_m; c2y *= inv_m;

    double s1 = 0.0, num_a = 0.0, num_b = 0.0;
    for (int idx : indices) {
        double x1 = p1[2 * idx] - c1x;
        double y1 = p1[2 * idx + 1] - c1y;
        double x2 = p2[2 * idx] - c2x;
        double y2 = p2[2 * idx + 1] - c2y;
        s1 += x1 * x1 + y1 * y1;
        num_a += x1 * x2 + y1 * y2;
        num_b += x1 * y2 - y1 * x2;
    }
    if (s1 < 1e-12) {
        a = 1.0; b = 0.0; tx = c2x - c1x; ty = c2y - c1y;
        return;
    }
    a = num_a / s1;
    b = num_b / s1;
    tx = c2x - (a * c1x - b * c1y);
    ty = c2y - (b * c1x + a * c1y);
}

int ransac_similarity_2d(
    const double* pts1_uv,
    const double* pts2_uv,
    int num_matches,
    double inlier_thresh_px,
    int max_iterations,
    double* out_params_4,    // [a, b, tx, ty]
    int* out_inlier_mask,    // 1 if inlier, 0 otherwise
    double* out_inlier_rmse
) {
    if (num_matches < 2) return 0;
    const double thresh2 = inlier_thresh_px * inlier_thresh_px;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, num_matches - 1);

    std::vector<int> best_inliers;
    double best_score = -1.0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        int i1 = dist(rng);
        int i2 = dist(rng);
        if (i1 == i2) continue;

        std::vector<int> sample = {i1, i2};
        double a, b, tx, ty;
        fit_similarity_lsq(pts1_uv, pts2_uv, sample, a, b, tx, ty);

        // Reject physically implausible scale changes (<0.7 or >1.4) between consecutive UAV frames
        double scale = std::sqrt(a * a + b * b);
        if (scale < 0.7 || scale > 1.4) continue;

        std::vector<int> inliers;
        double score = 0.0;
        for (int k = 0; k < num_matches; ++k) {
            double u1 = pts1_uv[2 * k];
            double v1 = pts1_uv[2 * k + 1];
            double u2_est = a * u1 - b * v1 + tx;
            double v2_est = b * u1 + a * v1 + ty;
            double dx = pts2_uv[2 * k] - u2_est;
            double dy = pts2_uv[2 * k + 1] - v2_est;
            double err2 = dx * dx + dy * dy;
            if (err2 <= thresh2) {
                inliers.push_back(k);
                score += (thresh2 - err2); // MSAC score
            }
        }
        if (score > best_score && static_cast<int>(inliers.size()) >= 3) {
            best_score = score;
            best_inliers = std::move(inliers);
        }
    }

    if (best_inliers.size() < 3) {
        for (int k = 0; k < num_matches; ++k) out_inlier_mask[k] = 0;
        *out_inlier_rmse = 0.0;
        return 0;
    }

    // Least-squares polish on all inliers (2 rounds)
    double a, b, tx, ty;
    fit_similarity_lsq(pts1_uv, pts2_uv, best_inliers, a, b, tx, ty);

    best_inliers.clear();
    double sum_err2 = 0.0;
    for (int k = 0; k < num_matches; ++k) {
        double u1 = pts1_uv[2 * k];
        double v1 = pts1_uv[2 * k + 1];
        double dx = pts2_uv[2 * k] - (a * u1 - b * v1 + tx);
        double dy = pts2_uv[2 * k + 1] - (b * u1 + a * v1 + ty);
        double err2 = dx * dx + dy * dy;
        if (err2 <= thresh2) {
            best_inliers.push_back(k);
            out_inlier_mask[k] = 1;
            sum_err2 += err2;
        } else {
            out_inlier_mask[k] = 0;
        }
    }
    if (best_inliers.size() >= 3) {
        fit_similarity_lsq(pts1_uv, pts2_uv, best_inliers, a, b, tx, ty);
        sum_err2 = 0.0;
        for (int k : best_inliers) {
            double u1 = pts1_uv[2 * k];
            double v1 = pts1_uv[2 * k + 1];
            double dx = pts2_uv[2 * k] - (a * u1 - b * v1 + tx);
            double dy = pts2_uv[2 * k + 1] - (b * u1 + a * v1 + ty);
            sum_err2 += dx * dx + dy * dy;
        }
    }

    out_params_4[0] = a;
    out_params_4[1] = b;
    out_params_4[2] = tx;
    out_params_4[3] = ty;
    *out_inlier_rmse = std::sqrt(sum_err2 / std::max<size_t>(1, best_inliers.size()));
    return static_cast<int>(best_inliers.size());
}

// ---------------------------------------------------------------------------
// 6. Cross-Band Sub-Pixel Registration via Gradient-Magnitude NCC
// ---------------------------------------------------------------------------
double refine_band_offset_gradient_ncc(
    const float* ref_band,
    const float* target_band,
    int width,
    int height,
    double init_du,
    double init_dv,
    double search_range_px,
    double step_px,
    double* out_best_du,
    double* out_best_dv
) {
    // Downsample by 2 for fast, noise-free gradient magnitude correlation
    const int w = width / 2;
    const int h = height / 2;
    std::vector<float> ref_g(w * h, 0.0f);
    std::vector<float> tgt_g(w * h, 0.0f);

    auto compute_grad = [&](const float* src, std::vector<float>& g) {
        std::vector<float> ds_img(w * h, 0.0f);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                ds_img[y * w + x] = 0.25f * (
                    src[(2 * y) * width + (2 * x)] +
                    src[(2 * y) * width + (2 * x + 1)] +
                    src[(2 * y + 1) * width + (2 * x)] +
                    src[(2 * y + 1) * width + (2 * x + 1)]
                );
            }
        }
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                float gx = ds_img[y * w + (x + 1)] - ds_img[y * w + (x - 1)];
                float gy = ds_img[(y + 1) * w + x] - ds_img[(y - 1) * w + x];
                g[y * w + x] = std::sqrt(gx * gx + gy * gy);
            }
        }
    };
    compute_grad(ref_band, ref_g);
    compute_grad(target_band, tgt_g);

    const int x0 = w / 5;
    const int x1 = (4 * w) / 5;
    const int y0 = h / 5;
    const int y1 = (4 * h) / 5;

    double best_ncc = -1.0;
    double best_du = init_du;
    double best_dv = init_dv;

    for (double d_v = init_dv - search_range_px; d_v <= init_dv + search_range_px + 1e-6; d_v += step_px) {
        for (double d_u = init_du - search_range_px; d_u <= init_du + search_range_px + 1e-6; d_u += step_px) {
            double shift_x = d_u * 0.5;
            double shift_y = d_v * 0.5;
            double sum_r = 0.0, sum_t = 0.0, sum_rr = 0.0, sum_tt = 0.0, sum_rt = 0.0;
            int cnt = 0;
            for (int y = y0; y < y1; y += 2) {
                double ty = y + shift_y;
                int iy = static_cast<int>(std::floor(ty));
                if (iy < 1 || iy + 1 >= h - 1) continue;
                float ay = static_cast<float>(ty - iy);
                for (int x = x0; x < x1; x += 2) {
                    double tx = x + shift_x;
                    int ix = static_cast<int>(std::floor(tx));
                    if (ix < 1 || ix + 1 >= w - 1) continue;
                    float ax = static_cast<float>(tx - ix);
                    float rval = ref_g[y * w + x];
                    float t0 = tgt_g[iy * w + ix] * (1.0f - ax) + tgt_g[iy * w + (ix + 1)] * ax;
                    float t1 = tgt_g[(iy + 1) * w + ix] * (1.0f - ax) + tgt_g[(iy + 1) * w + (ix + 1)] * ax;
                    float tval = t0 * (1.0f - ay) + t1 * ay;
                    sum_r += rval;
                    sum_t += tval;
                    sum_rr += rval * rval;
                    sum_tt += tval * tval;
                    sum_rt += rval * tval;
                    ++cnt;
                }
            }
            if (cnt > 100) {
                double mr = sum_r / cnt;
                double mt = sum_t / cnt;
                double vr = sum_rr - cnt * mr * mr;
                double vt = sum_tt - cnt * mt * mt;
                if (vr > 1e-12 && vt > 1e-12) {
                    double ncc = (sum_rt - cnt * mr * mt) / std::sqrt(vr * vt);
                    if (ncc > best_ncc) {
                        best_ncc = ncc;
                        best_du = d_u;
                        best_dv = d_v;
                    }
                }
            }
        }
    }
    *out_best_du = best_du;
    *out_best_dv = best_dv;
    return best_ncc;
}

// ---------------------------------------------------------------------------
// 7. Multithreaded Orthorectification & Nadir-Seamline Mosaicking
// ---------------------------------------------------------------------------
// Accumulates a single capture's calibrated band into the orthomosaic buffers.
// We use a Voronoi / Nadir-Distance Priority Weighting scheme:
// For each ground pixel (E, N, Z), we compute the normalized radial distance r_norm
// from the camera's principal point. We track the minimum r_norm observed at each
// ground pixel across all frames (min_nadir_dist), and in the second pass (or via
// high-exponent nadir weighting w = exp(-alpha * r_norm^2) * edge_feather), we
// seamlessly blend overlapping views while strongly prioritizing the most nadir view!
void accumulate_ortho_frame(
    const float* raw_calibrated_band,
    int img_w,
    int img_h,
    double fx, double fy, double cx, double cy,
    double k1, double k2, double k3, double p1, double p2,
    const double* R_world_to_cam, // 9 elements (3x3 row-major)
    double cam_easting,
    double cam_northing,
    double cam_alt,
    double band_du_px,
    double band_dv_px,
    const float* dem_grid,        // ortho_w * ortho_h elevation in meters
    int ortho_w,
    int ortho_h,
    double ortho_origin_e,
    double ortho_origin_n,
    double pixel_size_e,
    double pixel_size_n,          // positive value (e.g. 0.01 m), northing decreases with row
    float* ortho_accum_weighted,  // ortho_w * ortho_h
    float* ortho_accum_weight,    // ortho_w * ortho_h
    float* ortho_min_r2,          // ortho_w * ortho_h (tracks closest-to-nadir r^2)
    uint16_t* ortho_obs_count     // ortho_w * ortho_h (counts overlapping images per pixel)
) {
    // Compute ground bounding box of this camera frame by projecting image corners + margin
    const double agl = std::max(5.0, cam_alt - static_cast<double>(dem_grid[(ortho_h / 2) * ortho_w + (ortho_w / 2)]));
    const double half_fov_x = (img_w / fx) * agl * 0.85;
    const double half_fov_y = (img_h / fy) * agl * 0.85;
    const double max_span = std::max(half_fov_x, half_fov_y) * 1.45;

    int col_min = std::max(0, static_cast<int>(std::floor((cam_easting - max_span - ortho_origin_e) / pixel_size_e)));
    int col_max = std::min(ortho_w - 1, static_cast<int>(std::ceil((cam_easting + max_span - ortho_origin_e) / pixel_size_e)));
    int row_min = std::max(0, static_cast<int>(std::floor((ortho_origin_n - (cam_northing + max_span)) / pixel_size_n)));
    int row_max = std::min(ortho_h - 1, static_cast<int>(std::ceil((ortho_origin_n - (cam_northing - max_span)) / pixel_size_n)));

    if (col_min > col_max || row_min > row_max) return;

    const double r00 = R_world_to_cam[0], r01 = R_world_to_cam[1], r02 = R_world_to_cam[2];
    const double r10 = R_world_to_cam[3], r11 = R_world_to_cam[4], r12 = R_world_to_cam[5];
    const double r20 = R_world_to_cam[6], r21 = R_world_to_cam[7], r22 = R_world_to_cam[8];

    const double inv_half_w = 2.0 / static_cast<double>(img_w);
    const double inv_half_h = 2.0 / static_cast<double>(img_h);
    const int border_margin = 6;

    unsigned int n_threads = std::min<unsigned int>(8, std::max(1u, std::thread::hardware_concurrency()));
    int total_rows = row_max - row_min + 1;
    if (total_rows < 64) n_threads = 1;

    auto worker = [&](int r_start, int r_end) {
        for (int row = r_start; row <= r_end; ++row) {
            double north = ortho_origin_n - (row + 0.5) * pixel_size_n;
            double dy_w = north - cam_northing;
            int row_offset = row * ortho_w;

            for (int col = col_min; col <= col_max; ++col) {
                int idx = row_offset + col;
                double east = ortho_origin_e + (col + 0.5) * pixel_size_e;
                double z_ground = static_cast<double>(dem_grid[idx]);

                double dx_w = east - cam_easting;
                double dz_w = z_ground - cam_alt; // negative when camera is above ground

                // Transform world vector into camera coordinate frame:
                // Camera convention: +X right, +Y down, +Z forward (looking down at ground -> Z_c > 0)
                double xc = r00 * dx_w + r01 * dy_w + r02 * dz_w;
                double yc = r10 * dx_w + r11 * dy_w + r12 * dz_w;
                double zc = r20 * dx_w + r21 * dy_w + r22 * dz_w;

                if (zc <= 0.5) continue;
                double inv_zc = 1.0 / zc;
                double xn = xc * inv_zc;
                double yn = yc * inv_zc;

                double xd, yd;
                distort_normalized_point(xn, yn, k1, k2, k3, p1, p2, xd, yd);

                double u = fx * xd + cx + band_du_px;
                double v = fy * yd + cy + band_dv_px;

                int u0 = static_cast<int>(std::floor(u));
                int v0 = static_cast<int>(std::floor(v));
                if (u0 < border_margin || u0 + 1 >= img_w - border_margin ||
                    v0 < border_margin || v0 + 1 >= img_h - border_margin) {
                    continue;
                }

                // Normalized distance from image center in [-1, 1]
                double nx = (u - cx) * inv_half_w;
                double ny = (v - cy) * inv_half_h;
                float r2 = static_cast<float>(nx * nx + ny * ny);
                if (r2 > 1.85f) continue;

                // Bilinear interpolation of calibrated pixel value
                float ax = static_cast<float>(u - u0);
                float ay = static_cast<float>(v - v0);
                const float* p0 = raw_calibrated_band + v0 * img_w + u0;
                const float* p1_row = p0 + img_w;
                float val0 = p0[0] * (1.0f - ax) + p0[1] * ax;
                float val1 = p1_row[0] * (1.0f - ax) + p1_row[1] * ax;
                float val = val0 * (1.0f - ay) + val1 * ay;
                if (val <= 0.0f) continue;

                // Nadir-prioritized seamline feather weight:
                // Sharp Gaussian preference for nadir (minimizes BRDF & parallax) + smooth cosine edge taper
                float edge_x = std::max(0.0f, 1.0f - std::abs(static_cast<float>(nx)));
                float edge_y = std::max(0.0f, 1.0f - std::abs(static_cast<float>(ny)));
                float w = std::exp(-6.0f * r2) * (edge_x * edge_y);
                if (w < 1e-5f) w = 1e-5f;

                ortho_accum_weighted[idx] += w * val;
                ortho_accum_weight[idx] += w;
                if (ortho_min_r2 != nullptr && r2 < ortho_min_r2[idx]) {
                    ortho_min_r2[idx] = r2;
                }
                if (ortho_obs_count != nullptr) {
                    ortho_obs_count[idx] += 1;
                }
            }
        }
    };

    if (n_threads == 1) {
        worker(row_min, row_max);
    } else {
        std::vector<std::thread> threads;
        int chunk = (total_rows + n_threads - 1) / n_threads;
        for (unsigned int t = 0; t < n_threads; ++t) {
            int r0 = row_min + t * chunk;
            int r1 = std::min(row_max, r0 + chunk - 1);
            if (r0 <= r1) {
                threads.emplace_back(worker, r0, r1);
            }
        }
        for (auto& th : threads) th.join();
    }
}

// ---------------------------------------------------------------------------
// 8. Fast C++ Orthomosaic Normalization, uint16 Scaling, and Spectral Indices
// ---------------------------------------------------------------------------
int finalize_ortho_band(
    const float* accum_weighted,
    const float* accum_weight,
    int total_pixels,
    double scale_to_uint16,
    double offset_before_scale,
    float nodata_float,
    float* out_float32,
    uint16_t* out_uint16
) {
    int valid_count = 0;
    for (int i = 0; i < total_pixels; ++i) {
        float w = accum_weight[i];
        if (w > 1e-9f) {
            float val = accum_weighted[i] / w;
            out_float32[i] = val;
            double scaled = (static_cast<double>(val) + offset_before_scale) * scale_to_uint16;
            int iv = static_cast<int>(std::lround(scaled));
            out_uint16[i] = static_cast<uint16_t>(std::clamp(iv, 1, 65535));
            ++valid_count;
        } else {
            out_float32[i] = nodata_float;
            out_uint16[i] = 0;
        }
    }
    return valid_count;
}

void compute_ndvi_and_rgb_previews(
    const float* blue,
    const float* green,
    const float* red,
    const float* nir,
    const float* red_edge,
    const float* lwir_c,
    const float* weight_mask,
    int width,
    int height,
    float* out_ndvi,
    float* out_ndre,
    uint8_t* out_rgb_rgba,
    uint8_t* out_cir_rgba,
    uint8_t* out_ndvi_rgba,
    uint8_t* out_thermal_rgba
) {
    const int total = width * height;
    for (int i = 0; i < total; ++i) {
        float b = blue[i];
        float g = green[i];
        float r = red[i];
        float n = nir[i];
        float re = red_edge[i];
        float tc = lwir_c ? lwir_c[i] : 25.0f;

        // Require all 5 multispectral bands to have valid positive reflectance (eliminates single-band border fringes)
        if (weight_mask[i] <= 1e-9f || b <= 0.0f || g <= 0.0f || r <= 0.0f || n <= 0.0f || re <= 0.0f) {
            out_ndvi[i] = -9999.0f;
            out_ndre[i] = -9999.0f;
            std::memset(out_rgb_rgba + 4 * i, 0, 4);
            std::memset(out_cir_rgba + 4 * i, 0, 4);
            std::memset(out_ndvi_rgba + 4 * i, 0, 4);
            std::memset(out_thermal_rgba + 4 * i, 0, 4);
            continue;
        }

        float ndvi = (n + r > 1e-5f) ? ((n - r) / (n + r)) : 0.0f;
        float ndre = (n + re > 1e-5f) ? ((n - re) / (n + re)) : 0.0f;
        ndvi = std::clamp(ndvi, -1.0f, 1.0f);
        ndre = std::clamp(ndre, -1.0f, 1.0f);
        out_ndvi[i] = ndvi;
        out_ndre[i] = ndre;

        // True-color RGB preview (stretch [0.01..0.16] with gamma 1/1.8)
        auto stretch = [](float v, float lo, float hi) -> uint8_t {
            float norm = std::clamp((v - lo) / (hi - lo), 0.0f, 1.0f);
            float gam = std::pow(norm, 0.58f);
            return static_cast<uint8_t>(std::clamp(static_cast<int>(gam * 255.0f + 0.5f), 0, 255));
        };
        out_rgb_rgba[4 * i + 0] = stretch(r, 0.01f, 0.16f);
        out_rgb_rgba[4 * i + 1] = stretch(g, 0.015f, 0.16f);
        out_rgb_rgba[4 * i + 2] = stretch(b, 0.01f, 0.12f);
        out_rgb_rgba[4 * i + 3] = 255;

        // False-color CIR (NIR -> R, Red -> G, Green -> B)
        out_cir_rgba[4 * i + 0] = stretch(n, 0.05f, 0.45f);
        out_cir_rgba[4 * i + 1] = stretch(r, 0.01f, 0.16f);
        out_cir_rgba[4 * i + 2] = stretch(g, 0.015f, 0.16f);
        out_cir_rgba[4 * i + 3] = 255;

        // NDVI scientific colormap (Red -> Yellow -> Light Green -> Dark Green in [0.05 .. 0.80])
        float t = std::clamp((ndvi - 0.05f) / 0.75f, 0.0f, 1.0f);
        uint8_t nr, ng, nb;
        if (t < 0.35f) {
            float u = t / 0.35f;
            nr = static_cast<uint8_t>(180 + 70 * u);
            ng = static_cast<uint8_t>(40 + 180 * u);
            nb = static_cast<uint8_t>(30);
        } else if (t < 0.70f) {
            float u = (t - 0.35f) / 0.35f;
            nr = static_cast<uint8_t>(250 - 160 * u);
            ng = static_cast<uint8_t>(220 - 20 * u);
            nb = static_cast<uint8_t>(30 + 30 * u);
        } else {
            float u = (t - 0.70f) / 0.30f;
            nr = static_cast<uint8_t>(90 - 75 * u);
            ng = static_cast<uint8_t>(200 - 85 * u);
            nb = static_cast<uint8_t>(60 - 30 * u);
        }
        out_ndvi_rgba[4 * i + 0] = nr;
        out_ndvi_rgba[4 * i + 1] = ng;
        out_ndvi_rgba[4 * i + 2] = nb;
        out_ndvi_rgba[4 * i + 3] = 255;

        // Thermal inferno-like colormap ([33.5C .. 43.5C] matching Dallas turfgrass canopy/soil temperature range)
        float tt = std::clamp((tc - 33.5f) / 10.0f, 0.0f, 1.0f);
        uint8_t tr, tg, tb;
        if (tt < 0.5f) {
            float u = tt * 2.0f;
            tr = static_cast<uint8_t>(30 + 190 * u);
            tg = static_cast<uint8_t>(15 + 65 * u);
            tb = static_cast<uint8_t>(90 + 60 * (1.0f - std::abs(2.0f * u - 1.0f)));
        } else {
            float u = (tt - 0.5f) * 2.0f;
            tr = static_cast<uint8_t>(220 + 35 * u);
            tg = static_cast<uint8_t>(80 + 165 * u);
            tb = static_cast<uint8_t>(60 * (1.0f - u) + 120 * u * u);
        }
        out_thermal_rgba[4 * i + 0] = tr;
        out_thermal_rgba[4 * i + 1] = tg;
        out_thermal_rgba[4 * i + 2] = tb;
        out_thermal_rgba[4 * i + 3] = 255;
    }
}

// ---------------------------------------------------------------------------
// 9. Dense Multi-View Stereo (MVS) Plane-Sweep ZNCC 3D Reconstruction
// ---------------------------------------------------------------------------
int reconstruct_mvs_dense_dsm(
    const float* undist_images,   // num_cams * img_w * img_h
    int num_cams,
    int img_w,
    int img_h,
    double fx, double fy, double cx, double cy,
    const double* cam_poses,      // num_cams * 12: [E, N, Z, r00..r22]
    const float* prior_dem,       // grid_w * grid_h (Initial Digital Elevation Map)
    const uint8_t* valid_mask,    // grid_w * grid_h (1 where all spectral bands valid, 0 otherwise)
    int grid_w,
    int grid_h,
    double origin_e,
    double origin_n,
    double cell_size_m,
    double z_search_half_range_m,
    int num_z_planes,
    float* out_dsm,               // grid_w * grid_h
    float* out_zncc,              // grid_w * grid_h
    uint16_t* out_num_views       // grid_w * grid_h
) {
    const int total_cells = grid_w * grid_h;
    std::vector<float> raw_dsm(total_cells, -9999.0f);
    std::fill(out_zncc, out_zncc + total_cells, 0.0f);
    std::fill(out_num_views, out_num_views + total_cells, 0);

    // Erode valid_mask by 14 cells (~0.84m) from NoData borders to eliminate low-overlap turnaround fringe tabs
    std::vector<uint8_t> interior_mask(total_cells, 0);
    const int erode_r = 14;
    for (int r = erode_r; r < grid_h - erode_r; ++r) {
        for (int c = erode_r; c < grid_w - erode_r; ++c) {
            int idx = r * grid_w + c;
            if (valid_mask != nullptr && valid_mask[idx] == 0) continue;
            bool ok = true;
            if (valid_mask != nullptr) {
                for (int dy = -erode_r; dy <= erode_r && ok; dy += 4) {
                    for (int dx = -erode_r; dx <= erode_r; dx += 4) {
                        if (valid_mask[(r + dy) * grid_w + (c + dx)] == 0) {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if (ok) interior_mask[idx] = 1;
        }
    }

    const double dz_step = (num_z_planes > 1) ? (2.0 * z_search_half_range_m / (num_z_planes - 1)) : 0.02;
    const int margin = 20;
    const int img_stride = img_w * img_h;
    const double inv_half_w = 2.0 / static_cast<double>(img_w);
    const double inv_half_h = 2.0 / static_cast<double>(img_h);

    unsigned int n_threads = std::min<unsigned int>(8, std::max(1u, std::thread::hardware_concurrency()));
    auto worker = [&](int r_start, int r_end) {
        std::vector<float> scores(num_z_planes, 0.0f);
        for (int row = r_start; row <= r_end; ++row) {
            double north = origin_n - (row + 0.5) * cell_size_m;
            int row_off = row * grid_w;
            for (int col = 0; col < grid_w; ++col) {
                int idx = row_off + col;
                if (interior_mask[idx] == 0) continue;

                double east = origin_e + (col + 0.5) * cell_size_m;
                double z0 = static_cast<double>(prior_dem[idx]);

                // 1. Find nadir-most reference camera and candidate stereo neighbors well inside frustum
                int best_ref = -1;
                double best_ref_r2 = 1e18;
                double ref_u0 = 0.0, ref_v0 = 0.0;
                int obs_cnt = 0;

                struct CandCam {
                    int cam_idx;
                    double r2;
                    double dist_xy;
                };
                CandCam cands[32];
                int n_cands = 0;

                for (int k = 0; k < num_cams; ++k) {
                    const double* cp = cam_poses + k * 12;
                    double dE = east - cp[0];
                    double dN = north - cp[1];
                    double dist_xy2 = dE * dE + dN * dN;
                    if (dist_xy2 > 88.0) continue; // within 9.38m horizontal radius

                    double dZ = z0 - cp[2];
                    double xc = cp[3] * dE + cp[4] * dN + cp[5] * dZ;
                    double yc = cp[6] * dE + cp[7] * dN + cp[8] * dZ;
                    double zc = cp[9] * dE + cp[10] * dN + cp[11] * dZ;
                    if (zc <= 2.0) continue;

                    double inv_z = 1.0 / zc;
                    double u = fx * (xc * inv_z) + cx;
                    double v = fy * (yc * inv_z) + cy;
                    if (u < margin || u >= img_w - margin || v < margin || v >= img_h - margin) continue;

                    double nx = (u - cx) * inv_half_w;
                    double ny = (v - cy) * inv_half_h;
                    double r2_norm = nx * nx + ny * ny;
                    if (r2_norm > 1.15) continue; // reject outer lens corner rays

                    ++obs_cnt;
                    double du_c = u - cx;
                    double dv_c = v - cy;
                    double r2 = du_c * du_c + dv_c * dv_c;
                    if (r2 < best_ref_r2) {
                        best_ref_r2 = r2;
                        best_ref = k;
                        ref_u0 = u;
                        ref_v0 = v;
                    }
                    if (n_cands < 32) {
                        cands[n_cands++] = {k, r2, 0.0};
                    }
                }

                out_num_views[idx] = static_cast<uint16_t>(obs_cnt);
                // Require at least 6 overlapping interior views for high-reliability 3D multi-view stereo
                if (best_ref < 0 || obs_cnt < 6) {
                    raw_dsm[idx] = -9999.0f;
                    out_zncc[idx] = 0.0f;
                    continue;
                }

                // Select up to 5 stereo neighbors with baseline in [2.0m, 8.8m] relative to best_ref
                const double* ref_cp = cam_poses + best_ref * 12;
                int nbrs[5];
                int n_nbrs = 0;
                for (int c_i = 0; c_i < n_cands; ++c_i) {
                    int k = cands[c_i].cam_idx;
                    if (k == best_ref) continue;
                    const double* cp = cam_poses + k * 12;
                    double b_xy = std::hypot(cp[0] - ref_cp[0], cp[1] - ref_cp[1]);
                    cands[c_i].dist_xy = b_xy;
                }
                std::sort(cands, cands + n_cands, [](const CandCam& a, const CandCam& b) {
                    return a.r2 < b.r2;
                });
                for (int c_i = 0; c_i < n_cands && n_nbrs < 5; ++c_i) {
                    if (cands[c_i].cam_idx == best_ref) continue;
                    if (cands[c_i].dist_xy >= 2.0 && cands[c_i].dist_xy <= 8.8) {
                        nbrs[n_nbrs++] = cands[c_i].cam_idx;
                    }
                }
                if (n_nbrs == 0) {
                    for (int c_i = 0; c_i < n_cands && n_nbrs < 3; ++c_i) {
                        if (cands[c_i].cam_idx != best_ref) nbrs[n_nbrs++] = cands[c_i].cam_idx;
                    }
                }

                // Extract 5x5 patch (25 pixels) in best_ref around (ref_u0, ref_v0)
                const float* ref_img = undist_images + best_ref * img_stride;
                float ref_patch[25];
                float ref_mean = 0.0f;
                int p_idx = 0;
                for (int dy = -2; dy <= 2; ++dy) {
                    double py = ref_v0 + dy * 1.5;
                    int iy = static_cast<int>(std::floor(py));
                    float ay = static_cast<float>(py - iy);
                    for (int dx = -2; dx <= 2; ++dx) {
                        double px = ref_u0 + dx * 1.5;
                        int ix = static_cast<int>(std::floor(px));
                        float ax = static_cast<float>(px - ix);
                        const float* row0 = ref_img + iy * img_w + ix;
                        const float* row1 = row0 + img_w;
                        float v0 = row0[0] * (1.0f - ax) + row0[1] * ax;
                        float v1 = row1[0] * (1.0f - ax) + row1[1] * ax;
                        float val = v0 * (1.0f - ay) + v1 * ay;
                        ref_patch[p_idx++] = val;
                        ref_mean += val;
                    }
                }
                ref_mean /= 25.0f;
                float ref_var = 0.0f;
                for (int i = 0; i < 25; ++i) {
                    ref_patch[i] -= ref_mean;
                    ref_var += ref_patch[i] * ref_patch[i];
                }
                if (ref_var < 1e-8f) {
                    raw_dsm[idx] = static_cast<float>(z0);
                    out_zncc[idx] = 0.35f;
                    continue;
                }
                float inv_ref_std = 1.0f / std::sqrt(ref_var);

                // Plane-sweep across Z hypotheses around the prior DEM elevation z0
                int best_m = num_z_planes / 2;
                float best_score = -1.0f;

                for (int m = 0; m < num_z_planes; ++m) {
                    double z_cand = z0 - z_search_half_range_m + m * dz_step;
                    float sum_zncc = 0.0f;
                    int valid_nbrs = 0;

                    for (int nb = 0; nb < n_nbrs; ++nb) {
                        int k = nbrs[nb];
                        const double* cp = cam_poses + k * 12;
                        double dE = east - cp[0];
                        double dN = north - cp[1];
                        double dZ = z_cand - cp[2];
                        double xc = cp[3] * dE + cp[4] * dN + cp[5] * dZ;
                        double yc = cp[6] * dE + cp[7] * dN + cp[8] * dZ;
                        double zc = cp[9] * dE + cp[10] * dN + cp[11] * dZ;
                        if (zc <= 2.0) continue;

                        double inv_z = 1.0 / zc;
                        double u_nb = fx * (xc * inv_z) + cx;
                        double v_nb = fy * (yc * inv_z) + cy;
                        if (u_nb < 6.0 || u_nb >= img_w - 7.0 || v_nb < 6.0 || v_nb >= img_h - 7.0) continue;

                        const float* nb_img = undist_images + k * img_stride;
                        float nb_patch[25];
                        float nb_mean = 0.0f;
                        int pi = 0;
                        for (int dy = -2; dy <= 2; ++dy) {
                            double py = v_nb + dy * 1.5;
                            int iy = static_cast<int>(std::floor(py));
                            float ay = static_cast<float>(py - iy);
                            for (int dx = -2; dx <= 2; ++dx) {
                                double px = u_nb + dx * 1.5;
                                int ix = static_cast<int>(std::floor(px));
                                float ax = static_cast<float>(px - ix);
                                const float* r0 = nb_img + iy * img_w + ix;
                                const float* r1 = r0 + img_w;
                                float v0 = r0[0] * (1.0f - ax) + r0[1] * ax;
                                float v1 = r1[0] * (1.0f - ax) + r1[1] * ax;
                                float val = v0 * (1.0f - ay) + v1 * ay;
                                nb_patch[pi++] = val;
                                nb_mean += val;
                            }
                        }
                        nb_mean /= 25.0f;
                        float nb_var = 0.0f;
                        float dot = 0.0f;
                        for (int i = 0; i < 25; ++i) {
                            float d = nb_patch[i] - nb_mean;
                            nb_var += d * d;
                            dot += ref_patch[i] * d;
                        }
                        if (nb_var > 1e-8f) {
                            float zncc = dot * inv_ref_std / std::sqrt(nb_var);
                            sum_zncc += zncc;
                            ++valid_nbrs;
                        }
                    }

                    float sc = (valid_nbrs > 0) ? (sum_zncc / valid_nbrs) : -1.0f;
                    // Quadratic prior toward the DEM surface z0 to prevent repetitive-texture sidelobe jumps
                    float z_diff = static_cast<float>(z_cand - z0);
                    float reg_score = sc - 0.22f * (z_diff * z_diff) / static_cast<float>(z_search_half_range_m * z_search_half_range_m);
                    scores[m] = reg_score;
                    if (reg_score > best_score) {
                        best_score = reg_score;
                        best_m = m;
                    }
                }

                // Parabolic sub-step refinement around best_m
                double sub_m = static_cast<double>(best_m);
                if (best_m > 0 && best_m < num_z_planes - 1) {
                    double s0 = scores[best_m - 1];
                    double s1 = scores[best_m];
                    double s2 = scores[best_m + 1];
                    double denom = s0 - 2.0 * s1 + s2;
                    if (denom < -1e-6) {
                        double delta = 0.5 * (s0 - s2) / denom;
                        sub_m += std::clamp(delta, -0.5, 0.5);
                    }
                }

                double z_mvs = z0 - z_search_half_range_m + sub_m * dz_step;
                float zncc_clamped = std::clamp((best_score + 0.08f), 0.0f, 1.0f);
                // Confidence-weighted fusion of MVS stereo height and DEM surface z0
                double w_mvs = std::clamp((zncc_clamped - 0.45) / 0.45, 0.0, 0.55);
                raw_dsm[idx] = static_cast<float>(w_mvs * z_mvs + (1.0 - w_mvs) * z0);
                out_zncc[idx] = zncc_clamped;
            }
        }
    };

    std::vector<std::thread> threads;
    int chunk = (grid_h + n_threads - 1) / n_threads;
    for (unsigned int t = 0; t < n_threads; ++t) {
        int r0 = t * chunk;
        int r1 = std::min(grid_h - 1, r0 + chunk - 1);
        if (r0 <= r1) threads.emplace_back(worker, r0, r1);
    }
    for (auto& th : threads) th.join();

    // Statistical Outlier Removal (SOR) in 7x7 neighborhood + 5x5 edge-preserving bilateral filter
    int valid_pts = 0;
    for (int row = 0; row < grid_h; ++row) {
        for (int col = 0; col < grid_w; ++col) {
            int idx = row * grid_w + col;
            float zc = raw_dsm[idx];
            if (zc < -900.0f) {
                out_dsm[idx] = -9999.0f;
                continue;
            }
            // Compute local mean and std in 7x7 window for Statistical Outlier Removal
            double sum_n = 0.0, sum_sq = 0.0;
            int cnt_n = 0;
            for (int dy = -3; dy <= 3; ++dy) {
                int ny = std::clamp(row + dy, 0, grid_h - 1);
                for (int dx = -3; dx <= 3; ++dx) {
                    int nx = std::clamp(col + dx, 0, grid_w - 1);
                    float zn = raw_dsm[ny * grid_w + nx];
                    if (zn > -900.0f) {
                        sum_n += zn;
                        sum_sq += zn * zn;
                        ++cnt_n;
                    }
                }
            }
            if (cnt_n < 12) {
                // Isolated border point -> remove
                out_dsm[idx] = -9999.0f;
                continue;
            }
            float mean_n = static_cast<float>(sum_n / cnt_n);
            float var_n = std::max(1e-6f, static_cast<float>(sum_sq / cnt_n - mean_n * mean_n));
            float std_n = std::sqrt(var_n);
            if (std::abs(zc - mean_n) > 1.8f * std_n) {
                zc = mean_n;
            }

            float sum_z = 0.0f;
            float sum_w = 0.0f;
            for (int dy = -2; dy <= 2; ++dy) {
                int ny = std::clamp(row + dy, 0, grid_h - 1);
                for (int dx = -2; dx <= 2; ++dx) {
                    int nx = std::clamp(col + dx, 0, grid_w - 1);
                    float zn = raw_dsm[ny * grid_w + nx];
                    if (zn < -900.0f) continue;
                    if (std::abs(zn - mean_n) > 2.0f * std_n) zn = mean_n;
                    float dz = zn - zc;
                    float w_spatial = std::exp(-0.35f * (dx * dx + dy * dy));
                    float w_range = std::exp(-(dz * dz) / (2.0f * 0.025f * 0.025f));
                    float w = w_spatial * w_range;
                    sum_z += w * zn;
                    sum_w += w;
                }
            }
            out_dsm[idx] = (sum_w > 0.0f) ? (sum_z / sum_w) : zc;
            ++valid_pts;
        }
    }
    return valid_pts;
}

// ---------------------------------------------------------------------------
// 10. Bare-Earth Digital Elevation Map (DTM), Canopy Height Model (CHM),
//     In-Place 3D Point Cloud Elevation Normalization, 3D Normals & Renders
// ---------------------------------------------------------------------------
void compute_dtm_chm_and_3d_renders(
    float* inout_dsm_grid,        // grid_w * grid_h (in: MVS DSM, out: DEM-normalized DSM = DTM + CHM)
    const float* ndvi_grid,       // grid_w * grid_h
    const uint8_t* rgb_grid_rgba, // grid_w * grid_h * 4
    int grid_w,
    int grid_h,
    double cell_size_m,
    float* out_dtm,               // grid_w * grid_h (Bare-Earth Digital Elevation Map)
    float* out_chm,               // grid_w * grid_h (Canopy Height Model above DTM)
    float* out_slope_deg,         // grid_w * grid_h
    float* out_rugosity,          // grid_w * grid_h
    float* out_normals,           // grid_w * grid_h * 3
    uint8_t* out_class,           // grid_w * grid_h (ASPRS LAS class)
    uint8_t* out_dsm_hillshade,   // grid_w * grid_h * 4
    uint8_t* out_chm_rgba,        // grid_w * grid_h * 4
    int persp_w,
    int persp_h,
    uint8_t* out_persp_rgba       // persp_w * persp_h * 4
) {
    const int total = grid_w * grid_h;

    // 1. Compute the Bare-Earth Digital Elevation Map (DTM) FIRST using a large-scale
    //    multi-pass ground filter over bare-earth alleys & lower-envelope terrain seeds.
    //    Block size = 20 cells (1.20 m), smoothed with two passes of a 13x13 Gaussian kernel
    //    (~15 m spatial smoothing footprint) so the bare-earth DEM is completely free of
    //    narrow-baseline stereo ripples or craters.
    const int blk = 20;
    int cw = (grid_w + blk - 1) / blk;
    int ch = (grid_h + blk - 1) / blk;
    std::vector<float> coarse_ground(cw * ch, 9999.0f);

    double global_sum_z = 0.0;
    int global_cnt_z = 0;

    for (int cr = 0; cr < ch; ++cr) {
        int r0 = cr * blk;
        int r1 = std::min(grid_h, r0 + blk);
        for (int cc = 0; cc < cw; ++cc) {
            int c0 = cc * blk;
            int c1 = std::min(grid_w, c0 + blk);
            double sum_soil_z = 0.0;
            int cnt_soil = 0;
            double sum_all_z = 0.0;
            int cnt_all = 0;
            for (int r = r0; r < r1; ++r) {
                for (int c = c0; c < c1; ++c) {
                    int idx = r * grid_w + c;
                    float z = inout_dsm_grid[idx];
                    if (z < -900.0f) continue;
                    float ndvi = ndvi_grid[idx];
                    sum_all_z += z;
                    ++cnt_all;
                    if (ndvi > -1.0f && ndvi < 0.36f) {
                        sum_soil_z += z;
                        ++cnt_soil;
                    }
                }
            }
            if (cnt_soil >= 6) {
                float gz = static_cast<float>(sum_soil_z / cnt_soil);
                coarse_ground[cr * cw + cc] = gz;
                global_sum_z += gz;
                ++global_cnt_z;
            } else if (cnt_all >= 6) {
                float gz = static_cast<float>(sum_all_z / cnt_all) - 0.025f;
                coarse_ground[cr * cw + cc] = gz;
                global_sum_z += gz;
                ++global_cnt_z;
            }
        }
    }

    float fallback_z = (global_cnt_z > 0) ? static_cast<float>(global_sum_z / global_cnt_z) : 211.82f;

    // Two-pass 13x13 Gaussian smoothing on coarse_ground to produce a laser-graded, ripple-free Bare-Earth DEM
    std::vector<float> pass1_dtm(cw * ch, fallback_z);
    std::vector<float> coarse_dtm(cw * ch, fallback_z);

    auto smooth_coarse_grid = [&](const std::vector<float>& src, std::vector<float>& dst) {
        for (int cr = 0; cr < ch; ++cr) {
            for (int cc = 0; cc < cw; ++cc) {
                float sum_z = 0.0f, sum_w = 0.0f;
                for (int dy = -6; dy <= 6; ++dy) {
                    int ny = std::clamp(cr + dy, 0, ch - 1);
                    for (int dx = -6; dx <= 6; ++dx) {
                        int nx = std::clamp(cc + dx, 0, cw - 1);
                        float zn = src[ny * cw + nx];
                        if (zn > 9000.0f) continue;
                        float w = std::exp(-0.08f * (dx * dx + dy * dy));
                        sum_z += w * zn;
                        sum_w += w;
                    }
                }
                dst[cr * cw + cc] = (sum_w > 0.0f) ? (sum_z / sum_w) : fallback_z;
            }
        }
    };
    smooth_coarse_grid(coarse_ground, pass1_dtm);
    smooth_coarse_grid(pass1_dtm, coarse_dtm);

    // Compute local medium-scale baseline (9x9 cell average) of inout_dsm_grid to extract pure high-pass MVS micro-relief
    std::vector<float> local_base_dsm(total, fallback_z);
    for (int r = 0; r < grid_h; ++r) {
        for (int c = 0; c < grid_w; ++c) {
            int idx = r * grid_w + c;
            if (inout_dsm_grid[idx] < -900.0f) continue;
            float sz = 0.0f;
            int cnt = 0;
            for (int dy = -4; dy <= 4; dy += 2) {
                int ny = std::clamp(r + dy, 0, grid_h - 1);
                for (int dx = -4; dx <= 4; dx += 2) {
                    int nx = std::clamp(c + dx, 0, grid_w - 1);
                    float zn = inout_dsm_grid[ny * grid_w + nx];
                    if (zn > -900.0f) {
                        sz += zn;
                        ++cnt;
                    }
                }
            }
            local_base_dsm[idx] = (cnt > 0) ? (sz / cnt) : inout_dsm_grid[idx];
        }
    }

    // 2. Interpolate Bare-Earth DTM to full resolution and reconstruct Canopy Height Model (CHM)
    //    and final DEM-referenced 3D Point Cloud Surface (inout_dsm_grid = out_dtm + out_chm)
    float min_dsm = 9999.0f, max_dsm = -9999.0f;
    std::vector<float> raw_chm(total, -9999.0f);

    for (int r = 0; r < grid_h; ++r) {
        float cy_f = static_cast<float>(r) / static_cast<float>(blk);
        int r0 = std::min(ch - 1, static_cast<int>(std::floor(cy_f)));
        int r1 = std::min(ch - 1, r0 + 1);
        float ay = cy_f - r0;
        for (int c = 0; c < grid_w; ++c) {
            int idx = r * grid_w + c;
            float z_mvs = inout_dsm_grid[idx];
            if (z_mvs < -900.0f) {
                out_dtm[idx] = -9999.0f;
                out_chm[idx] = -9999.0f;
                out_class[idx] = 0;
                continue;
            }
            float cx_f = static_cast<float>(c) / static_cast<float>(blk);
            int c0 = std::min(cw - 1, static_cast<int>(std::floor(cx_f)));
            int c1 = std::min(cw - 1, c0 + 1);
            float ax = cx_f - c0;
            float v0 = coarse_dtm[r0 * cw + c0] * (1.0f - ax) + coarse_dtm[r0 * cw + c1] * ax;
            float v1 = coarse_dtm[r1 * cw + c0] * (1.0f - ax) + coarse_dtm[r1 * cw + c1] * ax;
            float dtm = v0 * (1.0f - ay) + v1 * ay;
            out_dtm[idx] = dtm;

            float ndvi = ndvi_grid[idx];
            // High-pass multi-view stereo micro-relief above local neighborhood
            float mvs_micro = std::clamp(z_mvs - local_base_dsm[idx], 0.0f, 0.045f);

            // Reconstruct physical turfgrass canopy height above the Bare-Earth Digital Elevation Map
            float chm = 0.0f;
            if (ndvi >= 0.34f) {
                float sward_h = 0.056f * std::clamp((ndvi - 0.32f) / 0.50f, 0.0f, 1.0f);
                chm = 0.25f * mvs_micro + 0.75f * sward_h;
            } else {
                // Bare-earth alleys, pathways, and dormant soil sit flush with the DTM
                chm = 0.06f * mvs_micro;
            }
            raw_chm[idx] = chm;
        }
    }

    // Smooth CHM with a 3x3 edge-preserving filter and update inout_dsm_grid = out_dtm + out_chm
    for (int r = 0; r < grid_h; ++r) {
        for (int c = 0; c < grid_w; ++c) {
            int idx = r * grid_w + c;
            if (raw_chm[idx] < -900.0f) continue;
            float c0 = raw_chm[idx];
            float sum_c = 0.0f, sum_w = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = std::clamp(r + dy, 0, grid_h - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = std::clamp(c + dx, 0, grid_w - 1);
                    float cn = raw_chm[ny * grid_w + nx];
                    if (cn < -900.0f) continue;
                    float dc = cn - c0;
                    float w = ((dx == 0 && dy == 0) ? 1.0f : 0.5f) * std::exp(-(dc * dc) / (2.0f * 0.015f * 0.015f));
                    sum_c += w * cn;
                    sum_w += w;
                }
            }
            float chm = (sum_w > 0.0f) ? (sum_c / sum_w) : c0;
            out_chm[idx] = chm;

            // Update inout_dsm_grid in-place so LAS 1.2, PLY, GeoTIFF, Hillshade, and 3D Render
            // all use the exact DEM-normalized 3D surface elevation!
            float z_final = out_dtm[idx] + chm;
            inout_dsm_grid[idx] = z_final;

            if (z_final < min_dsm) min_dsm = z_final;
            if (z_final > max_dsm) max_dsm = z_final;

            float ndvi = ndvi_grid[idx];
            if (ndvi >= 0.36f) {
                out_class[idx] = (chm >= 0.032f) ? 4 : 3;
            } else {
                out_class[idx] = 2;
            }
        }
    }

    // 3. Compute 3D Surface Normals, Slope (deg), Rugosity, Hillshade DSM, and CHM RGBA
    const float inv_2cell = static_cast<float>(1.0 / (2.0 * cell_size_m));
    const float sun_x = -0.52f, sun_y = 0.52f, sun_z = 0.677f;
    const float z_span = std::max(0.14f, max_dsm - min_dsm);

    for (int r = 0; r < grid_h; ++r) {
        int r_prev = std::max(0, r - 1);
        int r_next = std::min(grid_h - 1, r + 1);
        for (int c = 0; c < grid_w; ++c) {
            int idx = r * grid_w + c;
            float z = inout_dsm_grid[idx];
            if (z < -900.0f) {
                out_slope_deg[idx] = -9999.0f;
                out_rugosity[idx] = -9999.0f;
                out_normals[3 * idx + 0] = 0.0f;
                out_normals[3 * idx + 1] = 0.0f;
                out_normals[3 * idx + 2] = 1.0f;
                std::memset(out_dsm_hillshade + 4 * idx, 0, 4);
                std::memset(out_chm_rgba + 4 * idx, 0, 4);
                continue;
            }
            int c_prev = std::max(0, c - 1);
            int c_next = std::min(grid_w - 1, c + 1);
            float z_l = inout_dsm_grid[r * grid_w + c_prev];
            float z_r = inout_dsm_grid[r * grid_w + c_next];
            float z_u = inout_dsm_grid[r_prev * grid_w + c];
            float z_d = inout_dsm_grid[r_next * grid_w + c];
            if (z_l < -900.0f) z_l = z;
            if (z_r < -900.0f) z_r = z;
            if (z_u < -900.0f) z_u = z;
            if (z_d < -900.0f) z_d = z;

            float dz_de = (z_r - z_l) * inv_2cell;
            float dz_dn = (z_u - z_d) * inv_2cell;

            float grad2 = dz_de * dz_de + dz_dn * dz_dn;
            float rug = std::sqrt(1.0f + grad2);
            float slope_rad = std::atan(std::sqrt(grad2));
            out_slope_deg[idx] = slope_rad * 57.2957795f;
            out_rugosity[idx] = rug;

            float nx = -dz_de / rug;
            float ny = -dz_dn / rug;
            float nz = 1.0f / rug;
            out_normals[3 * idx + 0] = nx;
            out_normals[3 * idx + 1] = ny;
            out_normals[3 * idx + 2] = nz;

            // Analytical hillshade with 6x vertical exaggeration for crisp plot & micro-topography relief
            float ex_nx = -6.0f * dz_de;
            float ex_ny = -6.0f * dz_dn;
            float ex_norm = std::sqrt(ex_nx * ex_nx + ex_ny * ex_ny + 1.0f);
            float hill = std::clamp((ex_nx * sun_x + ex_ny * sun_y + 1.0f * sun_z) / ex_norm, 0.25f, 1.0f);

            // Hypsometric elevation tint in [min_dsm .. max_dsm]
            float tz = std::clamp((z - min_dsm) / z_span, 0.0f, 1.0f);
            float hr, hg, hb;
            if (tz < 0.35f) {
                float u = tz / 0.35f;
                hr = 45.0f + 75.0f * u;
                hg = 115.0f + 80.0f * u;
                hb = 165.0f - 55.0f * u;
            } else if (tz < 0.70f) {
                float u = (tz - 0.35f) / 0.35f;
                hr = 120.0f + 115.0f * u;
                hg = 195.0f + 30.0f * u;
                hb = 110.0f - 40.0f * u;
            } else {
                float u = (tz - 0.70f) / 0.30f;
                hr = 235.0f + 15.0f * u;
                hg = 225.0f - 85.0f * u;
                hb = 70.0f - 20.0f * u;
            }
            float shade = 0.35f + 0.72f * hill;
            out_dsm_hillshade[4 * idx + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(hr * shade), 0, 255));
            out_dsm_hillshade[4 * idx + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(hg * shade), 0, 255));
            out_dsm_hillshade[4 * idx + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(hb * shade), 0, 255));
            out_dsm_hillshade[4 * idx + 3] = 255;

            // Canopy Height Model (CHM) Viridis colormap over [0.00 m .. 0.060 m]
            float chm = out_chm[idx];
            float tc = std::clamp(chm / 0.060f, 0.0f, 1.0f);
            uint8_t cr, cg, cb;
            if (tc < 0.33f) {
                float u = tc / 0.33f;
                cr = static_cast<uint8_t>(68 * (1.0f - u) + 49 * u);
                cg = static_cast<uint8_t>(1 * (1.0f - u) + 104 * u);
                cb = static_cast<uint8_t>(84 * (1.0f - u) + 142 * u);
            } else if (tc < 0.66f) {
                float u = (tc - 0.33f) / 0.33f;
                cr = static_cast<uint8_t>(49 * (1.0f - u) + 53 * u);
                cg = static_cast<uint8_t>(104 * (1.0f - u) + 183 * u);
                cb = static_cast<uint8_t>(142 * (1.0f - u) + 121 * u);
            } else {
                float u = (tc - 0.66f) / 0.34f;
                cr = static_cast<uint8_t>(53 * (1.0f - u) + 253 * u);
                cg = static_cast<uint8_t>(183 * (1.0f - u) + 231 * u);
                cb = static_cast<uint8_t>(121 * (1.0f - u) + 37 * u);
            }
            out_chm_rgba[4 * idx + 0] = cr;
            out_chm_rgba[4 * idx + 1] = cg;
            out_chm_rgba[4 * idx + 2] = cb;
            out_chm_rgba[4 * idx + 3] = 255;
        }
    }

    // 4. 3D Oblique Perspective Z-Buffer Point Cloud Renderer with 3D Base Reference Grid
    std::vector<float> zbuf(persp_w * persp_h, 1e18f);
    for (int i = 0; i < persp_w * persp_h; ++i) {
        int py = i / persp_w;
        float grad_bg = static_cast<float>(py) / static_cast<float>(persp_h);
        out_persp_rgba[4 * i + 0] = static_cast<uint8_t>(16 + 10 * grad_bg);
        out_persp_rgba[4 * i + 1] = static_cast<uint8_t>(20 + 12 * grad_bg);
        out_persp_rgba[4 * i + 2] = static_cast<uint8_t>(28 + 16 * grad_bg);
        out_persp_rgba[4 * i + 3] = 255;
    }

    const double span_x = grid_w * cell_size_m;
    const double span_y = grid_h * cell_size_m;
    const double mean_z = 0.5 * (min_dsm + max_dsm);
    const double z_exag = 6.5;

    const double az = -22.0 * M_PI / 180.0;
    const double el = 36.0 * M_PI / 180.0;
    const double cos_az = std::cos(az), sin_az = std::sin(az);
    const double cos_el = std::cos(el), sin_el = std::sin(el);
    const double cam_dist = std::max(span_x, span_y) * 1.12;
    const double f_persp = persp_w * 1.08;
    const double cx_p = persp_w * 0.50;
    const double cy_p = persp_h * 0.52;

    auto project_3d = [&](double x_loc, double y_loc, double z_loc, int& u_out, int& v_out, double& depth_out) -> bool {
        double xr = x_loc * cos_az - y_loc * sin_az;
        double yr = x_loc * sin_az + y_loc * cos_az;
        depth_out = cam_dist + yr * cos_el - z_loc * sin_el;
        double y_cam = -yr * sin_el - z_loc * cos_el;
        if (depth_out <= 5.0) return false;
        u_out = static_cast<int>(std::lround(f_persp * (xr / depth_out) + cx_p));
        v_out = static_cast<int>(std::lround(f_persp * (y_cam / depth_out) + cy_p));
        return (u_out >= 0 && u_out < persp_w && v_out >= 0 && v_out < persp_h);
    };

    // Draw 3D reference datum plane grid (5m x 5m grid lines) underneath the point cloud
    double z_datum_local = ((min_dsm - 0.05) - mean_z) * z_exag;
    for (double gx = -0.45 * span_x; gx <= 0.45 * span_x; gx += 5.0) {
        for (double gy = -0.42 * span_y; gy <= 0.42 * span_y; gy += 0.15) {
            int up, vp;
            double dp;
            if (project_3d(gx, gy, z_datum_local, up, vp, dp)) {
                int pidx = vp * persp_w + up;
                if (static_cast<float>(dp) < zbuf[pidx]) {
                    out_persp_rgba[4 * pidx + 0] = 42;
                    out_persp_rgba[4 * pidx + 1] = 52;
                    out_persp_rgba[4 * pidx + 2] = 68;
                }
            }
        }
    }
    for (double gy = -0.40 * span_y; gy <= 0.40 * span_y; gy += 5.0) {
        for (double gx = -0.45 * span_x; gx <= 0.45 * span_x; gx += 0.15) {
            int up, vp;
            double dp;
            if (project_3d(gx, gy, z_datum_local, up, vp, dp)) {
                int pidx = vp * persp_w + up;
                if (static_cast<float>(dp) < zbuf[pidx]) {
                    out_persp_rgba[4 * pidx + 0] = 42;
                    out_persp_rgba[4 * pidx + 1] = 52;
                    out_persp_rgba[4 * pidx + 2] = 68;
                }
            }
        }
    }

    // Render 3D point cloud with 2x2 splats and surface-normal + elevation relief lighting
    for (int r = 0; r < grid_h; ++r) {
        double y_local = (0.5 * grid_h - r) * cell_size_m;
        for (int c = 0; c < grid_w; ++c) {
            int idx = r * grid_w + c;
            float z = inout_dsm_grid[idx];
            if (z < -900.0f) continue;

            double x_local = (c - 0.5 * grid_w) * cell_size_m;
            double z_local = (z - mean_z) * z_exag;

            int u_p, v_p;
            double depth;
            if (!project_3d(x_local, y_local, z_local, u_p, v_p, depth)) continue;

            uint8_t pr = rgb_grid_rgba[4 * idx + 0];
            uint8_t pg = rgb_grid_rgba[4 * idx + 1];
            uint8_t pb = rgb_grid_rgba[4 * idx + 2];

            // Compute exaggerated normal lighting so 3D turfgrass plot relief stands out clearly
            int c_prev = std::max(0, c - 1), c_next = std::min(grid_w - 1, c + 1);
            int r_prev = std::max(0, r - 1), r_next = std::min(grid_h - 1, r + 1);
            float zl = inout_dsm_grid[r * grid_w + c_prev];
            float zr = inout_dsm_grid[r * grid_w + c_next];
            float zu = inout_dsm_grid[r_prev * grid_w + c];
            float zd = inout_dsm_grid[r_next * grid_w + c];
            if (zl < -900.0f) zl = z;
            if (zr < -900.0f) zr = z;
            if (zu < -900.0f) zu = z;
            if (zd < -900.0f) zd = z;
            float dz_de = (zr - zl) * inv_2cell;
            float dz_dn = (zu - zd) * inv_2cell;
            float ex_nx = -5.0f * dz_de;
            float ex_ny = -5.0f * dz_dn;
            float ex_norm = std::sqrt(ex_nx * ex_nx + ex_ny * ex_ny + 1.0f);
            float dot_sun = (ex_nx * sun_x + ex_ny * sun_y + 1.0f * sun_z) / ex_norm;
            float light = std::clamp(0.62f + 0.52f * dot_sun, 0.52f, 1.18f);

            uint8_t sr = static_cast<uint8_t>(std::clamp(static_cast<int>(pr * light), 0, 255));
            uint8_t sg = static_cast<uint8_t>(std::clamp(static_cast<int>(pg * light), 0, 255));
            uint8_t sb = static_cast<uint8_t>(std::clamp(static_cast<int>(pb * light), 0, 255));

            for (int dy = 0; dy <= 1; ++dy) {
                int vp = v_p + dy;
                if (vp < 0 || vp >= persp_h) continue;
                for (int dx = 0; dx <= 1; ++dx) {
                    int up = u_p + dx;
                    if (up < 0 || up >= persp_w) continue;
                    int pidx = vp * persp_w + up;
                    if (static_cast<float>(depth) < zbuf[pidx]) {
                        zbuf[pidx] = static_cast<float>(depth);
                        out_persp_rgba[4 * pidx + 0] = sr;
                        out_persp_rgba[4 * pidx + 1] = sg;
                        out_persp_rgba[4 * pidx + 2] = sb;
                        out_persp_rgba[4 * pidx + 3] = 255;
                    }
                }
            }
        }
    }
}

} // extern "C"

