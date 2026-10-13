# Multispectral & 3D Photogrammetric Pipeline — Technical Report

## 1. Executive Summary & Generated Deliverables

A complete, end-to-end **Multispectral Orthomosaic** and **Dense Multi-View Stereo (MVS) 3D Point Cloud & 3D Canopy Metrics** pipeline was designed and implemented **from scratch** in an isolated `uv` environment using **pure Python standard library** modules (`struct`, `array`, `ctypes`, `math`, `zlib`, `csv`, `json`, `concurrent.futures`) coupled with a custom **C++17 native photogrammetric kernel** ([`photogrammetry_engine.cpp`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/src/ms_photogrammetry/cpp/photogrammetry_engine.cpp)). No external photogrammetry, computer vision, geospatial, point-cloud, or numerical packages (such as OpenCV, GDAL, Rasterio, laspy, PDAL, Open3D, COLMAP, OpenDroneMap, SciPy, or NumPy) were used.

### 1.1 Primary Orthomosaic Deliverables (`EPSG:32614`, `1.5 cm/px`)

| Deliverable | File Path | Format & CRS | Dimensions & Size | Spectral Content |
| :--- | :--- | :--- | :--- | :--- |
| **Primary 6-Band Georeferenced Orthomosaic (UInt16)** | [`multispectral_orthomosaic.tif`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/multispectral_orthomosaic.tif) & [`output/multispectral_orthomosaic_uint16.tif`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/multispectral_orthomosaic_uint16.tif) | GeoTIFF 1.1 (`EPSG:32614`, WGS 84 / UTM Zone 14N), `UInt16` (`NoData = 0`) | `6600 × 3394` px (`1.5 cm/px`, `256 MB`) | Band 1: Blue (`475 nm`), Band 2: Green (`560 nm`), Band 3: Red (`668 nm`), Band 4: RedEdge (`717 nm`), Band 5: NIR (`842 nm`) scaled $\times 10000$; Band 6: LWIR Thermal (`11000 nm`, centi-Kelvin) |
| **Scientific 6-Band Surface Reflectance Orthomosaic (Float32)** | [`output/multispectral_orthomosaic_float32.tif`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/multispectral_orthomosaic_float32.tif) | GeoTIFF 1.1 (`EPSG:32614`), `Float32` (`NoData = -9999`) | `6600 × 3394` px (`1.5 cm/px`, `513 MB`) | Bands 1–5: Calibrated Surface Reflectance $\rho \in [0, 1]$; Band 6: LWIR Surface Temperature ($^\circ\text{C}$) |
| **Spectral Vegetation Indices (NDVI & NDRE)** | [`output/multispectral_indices_ndvi_ndre.tif`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/multispectral_indices_ndvi_ndre.tif) | GeoTIFF 1.1 (`EPSG:32614`), `Float32` (`NoData = -9999`) | `6600 × 3394` px (`1.5 cm/px`, `171 MB`) | Band 1: $\text{NDVI}$, Band 2: $\text{NDRE}$ |

### 1.2 3D Photogrammetric Point Cloud & 3D Metrics Deliverables (`EPSG:32614`, `1,058,753` 3D Points)

| Deliverable | File Path | Format & CRS | Point Count / Grid | 3D & Spectral Attributes |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Dense 3D Point Cloud (ASPRS LAS 1.2)** | [`photogrammetric_point_cloud.las`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/photogrammetric_point_cloud.las) & [`output/dense_photogrammetric_point_cloud.las`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/dense_photogrammetric_point_cloud.las) | ASPRS `LAS 1.2` Binary (Record Format 2, $0.1\text{ mm}$ coordinate precision, `EPSG:32614`) | `1,058,753` points (`277.8 pts/m²`, `28 MB`) | `X, Y, Z` (UTM 14N), `Intensity` (Calibrated NIR), `Classification` (`2=Ground`, `3=Low Turf`, `4=Dense Turf`, `6=Structure`), 16-bit `R, G, B`, `UserData` (Stereo Views) |
| **Multi-Attribute Scientific 3D Point Cloud (PLY)** | [`output/dense_photogrammetric_point_cloud.ply`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/dense_photogrammetric_point_cloud.ply) | Stanford `PLY` Binary Little-Endian (`EPSG:32614`) | `1,058,753` vertices (`53 MB`) | `x, y, z`, 3D Surface Normals `nx, ny, nz`, `red, green, blue`, `canopy_height_m`, `ndvi`, `ndre`, `temperature_c`, `zncc_confidence`, `classification` |
| **Sparse SfM 3D Tie-Point Cloud (PLY & CSV)** | [`output/sparse_sfm_tiepoints.ply`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/sparse_sfm_tiepoints.ply) & [`output/sparse_sfm_tiepoints.csv`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/sparse_sfm_tiepoints.csv) | Binary `PLY` & `CSV` (`EPSG:32614`) | `6,000` multi-view tie points | Triangulated 3D tracks, observation multiplicity, reprojection error |
| **5-Band 3D Metric Raster GeoTIFF (DSM / DTM / CHM / Slope / Rugosity)** | [`output/photogrammetric_3d_dsm_dtm_chm.tif`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/photogrammetric_3d_dsm_dtm_chm.tif) | GeoTIFF 1.1 (`EPSG:32614`), `Float32` (`NoData = -9999`) | `1650 × 846` px (`6.0 cm/px`, `27 MB`) | Band 1: `DSM_Elevation_m`, Band 2: `DTM_Bare_Earth_m`, Band 3: `CHM_Canopy_Height_m`, Band 4: `Surface_Slope_deg`, Band 5: `Surface_Rugosity_3D_2D_Ratio` |
| **Per-Plot 3D Biovolume & Canopy Height Table** | [`output/plot_level_3d_metrics.csv`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/plot_level_3d_metrics.csv) | CSV Table | `490` segmented $2\text{ m} \times 2\text{ m}$ turf plots | Per-plot 3D Canopy Height (mean, P90, max), 3D Biovolume ($\text{cm}^3/\text{m}^2$), Rugosity, Slope, FGCC, NDVI, NDRE, LWIR Temp |
| **3D Point Cloud Metrics JSON Report** | [`output/point_cloud_3d_metrics_report.json`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/point_cloud_3d_metrics_report.json) | JSON | Field-wide & plot summary | Complete 3D DSM, DTM, CHM, Biovolume, Rugosity, and ASPRS class metrics |
| **3D Visual Renderings (Perspective, DSM Hillshade, CHM)** | [`output/point_cloud_3d_perspective_render.png`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/point_cloud_3d_perspective_render.png), [`output/point_cloud_dsm_hillshade.png`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/point_cloud_dsm_hillshade.png), [`output/point_cloud_chm_map.png`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/point_cloud_chm_map.png) | 32-bit RGBA PNG | `1600 × 1000` & `1650 × 846` px | Z-buffer oblique 3D perspective render, Lambertian hillshaded DSM, and Viridis CHM map |

---

## 2. Visual Inspection Gallery (2D Orthomosaics & 3D Point Cloud Products)

````carousel
![3D Oblique Perspective Z-Buffer Point Cloud Rendering (1,058,753 3D Points)](/Users/haruiz/.gemini/antigravity-ide/brain/557696b8-18c2-4b8c-ab60-415c4d771262/point_cloud_3d_perspective_render.png)
<!-- slide -->
![Photogrammetric Digital Surface Model (DSM) with Analytical Lambertian Hillshading](/Users/haruiz/.gemini/antigravity-ide/brain/557696b8-18c2-4b8c-ab60-415c4d771262/point_cloud_dsm_hillshade.png)
<!-- slide -->
![3D Canopy Height Model (CHM = DSM - Bare-Earth DTM, 0.0 to 7.5 cm)](/Users/haruiz/.gemini/antigravity-ide/brain/557696b8-18c2-4b8c-ab60-415c4d771262/point_cloud_chm_map.png)
<!-- slide -->
![True-Color RGB Composite Orthomosaic (Bands 3, 2, 1)](/Users/haruiz/.gemini/antigravity-ide/brain/557696b8-18c2-4b8c-ab60-415c4d771262/orthomosaic_rgb_preview.png)
<!-- slide -->
![Color-Infrared (CIR) False-Color Orthomosaic (NIR, Red, Green)](/Users/haruiz/.gemini/antigravity-ide/brain/557696b8-18c2-4b8c-ab60-415c4d771262/orthomosaic_cir_false_color.png)
<!-- slide -->
![Normalized Difference Vegetation Index (NDVI) Map](/Users/haruiz/.gemini/antigravity-ide/brain/557696b8-18c2-4b8c-ab60-415c4d771262/orthomosaic_ndvi_map.png)
<!-- slide -->
![LWIR Thermal Surface Temperature Map (33.5°C – 43.5°C)](/Users/haruiz/.gemini/antigravity-ide/brain/557696b8-18c2-4b8c-ab60-415c4d771262/orthomosaic_lwir_thermal.png)
<!-- slide -->
![Bundle-Adjusted Flight Trajectory (216 Poses) & 6,000 3D Tie Points](/Users/haruiz/.gemini/antigravity-ide/brain/557696b8-18c2-4b8c-ab60-415c4d771262/trajectory_and_tiepoints.png)
````

---

## 3. Dataset Inspection & Sensor Characterization (Stage 1)

### 3.1 Sensor Architecture (`MicaSense Altum-PT`)
The custom binary TIFF 6.0 / EXIF / GPS / XMP parser ([`src/ms_photogrammetry/tiff_io.py`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/src/ms_photogrammetry/tiff_io.py)) inspected all **1,666 raw TIFF files** in `Images/multispectral/`, grouping them into **238 synchronized 7-band capture events** (`IMG_0000` through `IMG_0237`):

| Band Index | Spectral Band | Central Wavelength | Bandwidth (FWHM) | Sensor Dimensions | Focal Length ($f_x = f_y$) | Principal Point $(c_x, c_y)$ | Rig Translation vs Band 2 $(\Delta X, \Delta Y, \Delta Z)$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Blue** | $475.0\text{ nm}$ | $32.0\text{ nm}$ | $2064 \times 1544$ | $7.877\text{ mm}$ ($2283.24\text{ px}$) | $(1034.63, 757.58)\text{ px}$ | $(-49.30, +0.24, +0.87)\text{ mm}$ |
| **2** | **Green (Ref)** | $560.0\text{ nm}$ | $27.0\text{ nm}$ | $2064 \times 1544$ | $7.878\text{ mm}$ ($2283.48\text{ px}$) | $(1040.98, 758.31)\text{ px}$ | $(0.00, 0.00, 0.00)\text{ mm}$ |
| **3** | **Red** | $668.0\text{ nm}$ | $14.0\text{ nm}$ | $2064 \times 1544$ | $7.859\text{ mm}$ ($2277.97\text{ px}$) | $(1046.10, 755.97)\text{ px}$ | $(-12.18, +24.87, -0.69)\text{ mm}$ |
| **4** | **NIR** | $842.0\text{ nm}$ | $57.0\text{ nm}$ | $2064 \times 1544$ | $7.888\text{ mm}$ ($2286.31\text{ px}$) | $(1039.12, 755.60)\text{ px}$ | $(-24.62, +0.30, -0.89)\text{ mm}$ |
| **5** | **RedEdge** | $717.0\text{ nm}$ | $12.0\text{ nm}$ | $2064 \times 1544$ | $7.894\text{ mm}$ ($2288.16\text{ px}$) | $(1045.42, 772.20)\text{ px}$ | $(-37.05, +24.91, -0.58)\text{ mm}$ |
| **6** | **Panchro** | $634.5\text{ nm}$ | $463.0\text{ nm}$ | $4112 \times 3008$ | $16.657\text{ mm}$ ($4828.00\text{ px}$) | $(2000.77, 1490.73)\text{ px}$ | $(-37.07, -47.48, +5.82)\text{ mm}$ |
| **7** | **LWIR Thermal** | $11000.0\text{ nm}$ | $6000.0\text{ nm}$ | $320 \times 256$ | $4.481\text{ mm}$ ($373.46\text{ px}$) | $(138.09, 122.47)\text{ px}$ | $(-7.37, -50.78, -6.95)\text{ mm}$ |

### 3.2 Flight Geometry, Footprint & Overlap
- **Active Lawnmower Survey Captures (`216`)**: `IMG_0009` through `IMG_0224`, organized into **12 East–West flight lines** at a mean altitude of **$230.98\text{ m}$ MSL** (**$19.08\text{ m}$ AGL** above the $211.90\text{ m}$ ground datum).
- **Native Multispectral GSD**: $0.836\text{ cm/px}$ ($17.25\text{ m} \times 12.90\text{ m}$ footprint), with **$80.27\%$ forward overlap** ($3.40\text{ m}$ spacing) and **$74.59\%$ side overlap** ($3.28\text{ m}$ line spacing).

---

## 4. Photogrammetric & Radiometric Mathematical Formulations (Stages 2–8)

1. **Radiometric Calibration & Vignetting Compensation** ([`src/ms_photogrammetry/radiometry.py`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/src/ms_photogrammetry/radiometry.py)):
   Converts raw 16-bit digital numbers $\text{DN}(u,v)$ to dark-subtracted, 2D rational-polynomial vignette-corrected spectral radiance $L(u,v)$ and transfers to surface reflectance $\rho_b(u,v) \in [0, 1]$ using pre/post-flight calibrated reflectance panels (`IMG_0002`, `IMG_0236`, serial `RP06-2315088-OB`) and time-varying DLS2 solar irradiance.
2. **Structure-from-Motion (SfM) & Global Bundle Adjustment** ([`src/ms_photogrammetry/sfm.py`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/src/ms_photogrammetry/sfm.py)):
   Connects **100% (`216 / 216`)** of survey captures across **931 RANSAC-verified stereo pairs** and jointly refines camera exterior orientations $(E_i, N_i, Z_i, \psi_i)$ and **6,000 3D tie points**, reducing reprojection RMSE from **$22.62\text{ px}$ to $14.04\text{ px}$** (median tie-point reprojection error **$6.44\text{ px} \approx 5.4\text{ cm}$** on the ground).
3. **Cross-Band Parallax & Boresight Registration** ([`src/ms_photogrammetry/registration.py`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/src/ms_photogrammetry/registration.py)):
   Aligns all 6 spectral/thermal channels to Reference Band 2 (Green) via C++ Gradient-Magnitude Normalized Cross-Correlation (GM-NCC) and 3D lever-arm baseline compensation (`0.89 px – 2.47 px` agreement with factory rig calibration).

---

## 5. DEM-First Dense Multi-View Stereo (MVS) 3D Point Cloud & 3D Metrics Pipeline (Stages 5 & 8)

Implemented from scratch in [`src/ms_photogrammetry/orthorectification.py`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/src/ms_photogrammetry/orthorectification.py), [`src/ms_photogrammetry/point_cloud.py`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/src/ms_photogrammetry/point_cloud.py), and C++17 kernels `reconstruct_mvs_dense_dsm` and `compute_dtm_chm_and_3d_renders` ([`photogrammetry_engine.cpp`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/src/ms_photogrammetry/cpp/photogrammetry_engine.cpp)):

### 5.1 Continuous Digital Elevation Map (DEM) Construction First (Stage 5)
1. **MAD Outlier-Filtered Tie-Point Terrain Fitting (`build_dem_grid`)**:
   Before orthorectification or dense canopy reconstruction, multi-view 3D tie points ($\ge 3$ observations) are filtered via Median Absolute Deviation (MAD) to reject narrow-baseline ray triangulation outliers, fit to a global ridge-regularized planar drainage grade $Z_{\text{trend}}(E, N) = Z_{\text{med}} + b_E(E - E_c) + b_N(N - N_c)$, and interpolated with a wide Gaussian kernel ($\sigma = 14\text{ m}$). This eliminates artificial perimeter step-cliffs where camera coverage drops off at the field boundary.
2. **Multi-Scale Bare-Earth Digital Terrain Model ($Z_{\text{DTM}}$)**:
   In turfgrass field trials, bare-soil alleys ($\text{NDVI} < 0.36$) surround the vegetated turf plots. The C++ engine extracts bare-soil ground seeds across $1.20\text{ m} \times 1.20\text{ m}$ blocks (`20 × 20` cells) and applies a two-pass $13 \times 13$ Gaussian terrain filter ($\sim 15\text{ m}$ effective footprint) to construct a ripple-free, laser-graded Bare-Earth Digital Elevation Map ([`digital_elevation_map.tif`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/digital_elevation_map.tif) and [`output/digital_elevation_map_dem.tif`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/digital_elevation_map_dem.tif)).

### 5.2 Multi-View Plane-Sweep ZNCC Dense Reconstruction & DEM-Normalized 3D Surface
1. **Strict Interior Multi-View Support ($\ge 6$ Overlapping Views)**:
   For each 3D ground post $(E, N)$ on a **$6.0\text{ cm}$ grid** (`1650 × 846` cells, **$277.78\text{ points/m}^2$**), the C++ engine erodes the outer boundary by $14$ cells ($0.84\text{ m}$) and requires at least **$6$ interior camera observations** (`obs_cnt >= 6`, mean **$13.16$ views/point**) well inside the lens frustum. This completely eliminates low-overlap turnaround fringe tabs and road-edge noise.
2. **5×5 Patch Plane Sweep, Parabolic Refinement & Statistical Outlier Removal (SOR)**:
   Across 17 candidate elevation planes $Z_m \in [Z_0 - 0.14\text{ m}, \, Z_0 + 0.14\text{ m}]$ around the DEM surface $Z_0$, the engine evaluates $5 \times 5$ patch **Zero-Mean Normalized Cross-Correlation (ZNCC)** across up to 5 stereo neighbor cameras ($B_{i,j} \in [2.0\text{ m}, 8.8\text{ m}]$), followed by parabolic sub-step refinement, $7 \times 7$ Statistical Outlier Removal ($|Z - \mu_Z| \le 1.8\sigma_Z$), and $5 \times 5$ bilateral filtering.
3. **In-Place DEM-Normalized Point Cloud Elevation ($Z_{\text{DSM}} = Z_{\text{DTM}} + h_{\text{CHM}}$)**:
   Canopy height $h_{\text{CHM}}(E, N) \ge 0$ is reconstructed directly above the Bare-Earth Digital Elevation Map $Z_{\text{DTM}}(E, N)$ and written in-place to `inout_dsm_grid` so that all exported `LAS 1.2` and `PLY` 3D point clouds sit cleanly on the continuous Digital Elevation Map without underground pits or perimeter steps.

### 5.3 Quantitative 3D Point Cloud & Canopy Metrics Summary
| 3D Metric Category | Parameter | Value |
| :--- | :--- | :--- |
| **Point Cloud Geometry** | **Total Dense MVS 3D Points (`LAS 1.2` & `PLY`)** | **`735,614` interior points** (`6,000` sparse SfM tie points) |
| | **Post Spacing / Point Density** | **`6.0 cm`** (**`277.78 points/m²`**) |
| | **Mean Stereo Views per 3D Point** | **`13.16` views** (minimum $\ge 6$ views enforced) |
| **3D Surface Area & Volume** | **2D Planar Footprint Area** | **`2,648.21 m²`** |
| | **3D True Surface Area** | **`2,654.82 m²`** (Overall Rugosity Ratio: **`1.00250`**) |
| | **Total Green Canopy 3D Biovolume** | **`39.005 m³`** |
| **Elevation & Canopy Height** | **DSM Surface Elevation (`EPSG:32614` MSL)** | Min: `211.679 m` \| Median: **`211.834 m`** \| Max: `212.022 m` |
| | **Bare-Earth DEM / DTM Elevation (MSL)** | Min: `211.671 m` \| Median: **`211.818 m`** \| Max: `211.989 m` |
| | **Vegetated Turfgrass Canopy Height (`CHM`)** | Mean: **`2.64 cm`** \| Median: **`2.84 cm`** \| P90: **`4.09 cm`** \| Max: **`4.46 cm`** |
| **ASPRS Point Classification** | `Class 2` (Bare Ground / Inter-Plot Alleys) | `325,044` points (`44.2%`) |
| | `Class 3` (Low Turfgrass Canopy `< 3.2 cm`) | `242,062` points (`32.9%`) |
| | `Class 4` (Dense/Tall Turfgrass Canopy `>= 3.2 cm`) | `168,508` points (`22.9%`) |
| **Plot-Level 3D Analytics** | **Segmented $2\text{ m} \times 2\text{ m}$ Evaluation Plots** | **`472` plots** exported to [`output/plot_level_3d_metrics.csv`](file:///Users/haruiz/Library/CloudStorage/GoogleDrive-henry.ruiz@tamu.edu/.shortcut-targets-by-id/1Z_PrOSlqpYJgJ_czaPNSCxUmXSOCoeN3/Dallas%20Turfgrass%20Drone%20images_2026/042026_drone%20flight_Nav/output/plot_level_3d_metrics.csv) |

---

## 6. Reproducibility & Verification Commands

```bash
# 1. Run the automated unit & integration test suite (8 tests including LAS 1.2 & PLY writers)
uv run --no-sync python -m unittest discover -s tests -v

# 2. Run the complete 9-stage multispectral orthomosaic + 3D point cloud pipeline
uv run --no-sync python -m ms_photogrammetry.cli \
  --image-dir "Images/multispectral" \
  --output-dir "output" \
  --resolution-m 0.015 \
  --point-spacing-m 0.06
```
