# Idol Q8 fidelity benchmark

Status: **COMPLETE_MEASUREMENTS**. Objects: 32/32; matched cases: 96/96.

A household-object geometry challenge cohort specified before inference, not a random population sample. Three seeds are clustered within each object. Original-versus-original seed variation changes the whole stochastic pipeline; it is context, not an error allowance or a subtraction from quantization error. The intervals are empirical stability intervals from 2,000 stratified object resamples, not population coverage guarantees. Ground-truth 3D reconstruction accuracy is not established. Approximate orthographic rendering is separate from upstream rendering and target-image accuracy. No quality threshold was chosen after seeing results.

| Measure | Valid matched cases | Mean | Median | 95th percentile | Range | Original seed-baseline mean |
|---|---:|---:|---:|---:|---:|---:|
| Mean mesh distance / reference diagonal (%) | 96 | 0.12368 | 0.089172 | 0.22064 | 0.032214–1.7151 | 2.4799 |
| Directional 95th-percentile mesh distance / diagonal (%) | 96 | 0.37891 | 0.25965 | 0.73045 | 0.082513–5.6093 | 7.1395 |
| Maximum sampled mesh distance / diagonal (%) | 96 | 2.8552 | 2.4154 | 7.0442 | 0.25278–10.022 | 11.361 |
| Mesh F1 within 0.1% of reference diagonal (%) | 96 | 68.487 | 71.51 | 95.593 | 5.239–97.487 | 8.8345 |
| Mesh F1 within 0.5% of reference diagonal (%) | 96 | 96.744 | 98.921 | 100 | 24.783–100 | 34.61 |
| Mean foreground color absolute error (%) | 96 | 2.2188 | 1.6844 | 4.1688 | 0.46017–28.586 | 18.163 |
| Mean silhouette overlap (%) | 96 | 98.633 | 99.38 | 99.896 | 64.697–99.943 | 76.694 |
| Relative latent Euclidean error (%) | 96 | 10.728 | 7.647 | 24.739 | 3.0711–78.242 | 85.523 |

## Strata

| Geometry/use stratum | Matched cases | Mean mesh error (%) | Tight F1 mean (%) | Foreground color error (%) |
|---|---:|---:|---:|---:|
| devices_and_appliances | 12 | 0.13836 | 63.047 | 2.138 |
| footwear | 12 | 0.1191 | 64.893 | 2.3425 |
| handles_and_cavities | 12 | 0.12115 | 61.801 | 1.0576 |
| irregular_and_multipart | 12 | 0.23356 | 64.303 | 4.6454 |
| printed_packaging | 12 | 0.11027 | 66.655 | 2.2775 |
| round_and_cylindrical | 12 | 0.096572 | 71.476 | 1.3021 |
| soft_and_folded | 12 | 0.12193 | 65.767 | 2.3128 |
| thin_and_flat | 12 | 0.048461 | 89.953 | 1.6745 |

## Empirical stability intervals

Each object is one resampling unit, with seeds retained together. Objects are resampled within their declared strata. These intervals describe stability of this selected cohort; they are not population coverage guarantees.

| Measure | 95% resampling interval for cohort mean | Paired mean difference from original seed baseline |
|---|---:|---:|
| Mean mesh distance / reference diagonal (%) | 0.096968–0.1584 | -3.2361–-1.5721 |
| Directional 95th-percentile mesh distance / diagonal (%) | 0.28981–0.49576 | -8.8555–-4.8004 |
| Maximum sampled mesh distance / diagonal (%) | 2.3863–3.3393 | -10.866–-6.2212 |
| Mesh F1 within 0.1% of reference diagonal (%) | 64.511–72.14 | 55.974–63.286 |
| Mesh F1 within 0.5% of reference diagonal (%) | 95.064–98.104 | 56.929–67.201 |
| Mean foreground color absolute error (%) | 1.7077–2.8699 | -18.37–-13.605 |
| Mean silhouette overlap (%) | 97.829–99.185 | 17.603–26.66 |
| Relative latent Euclidean error (%) | 8.9069–12.769 | -79.399–-69.87 |

## Exploratory preprocessing and camera diagnostics

These analyses were added during the run; they are not part of the analysis specified before inference.

| Diagnostic | Valid cases | Mean | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|---:|
| Preprocessed input channel absolute error (%) | 96 | 0.018224 | 0.0026079 | 0.17969 | 0.18042 |
| Preprocessed input pixels with any channel change (%) | 96 | 2.235 | 0.75908 | 14.641 | 16.812 |
| Predicted camera yaw difference (degrees) | 96 | 0.59859 | 0.29891 | 2.2139 | 10.579 |
| Predicted camera pitch difference (degrees) | 96 | 0.25427 | 0.18076 | 0.54338 | 3.6125 |
| Predicted vertical field-of-view difference (degrees) | 96 | 0.71157 | 0.57698 | 1.535 | 6.1286 |
| Predicted camera radius relative difference (%) | 96 | 1.4246 | 1.0774 | 2.9823 | 8.5329 |

## Failures

0 recorded failure entries; generation and derived comparison failures may refer to the same failed inference.
{"successful": 196, "failed_attempts": 0, "not_attempted_after_session_failure": 0}

## Same-seed repeatability

[
  {
    "object": "ASICS_GELLinksmaster_WhiteSilverCarolina_Blue",
    "identical_ply_bytes": true,
    "relative_latent_error_percent": 0.0,
    "foreground_color_error_percent": 0.0,
    "silhouette_overlap_percent": 100.0,
    "mesh_mean_error_percent": 2.3953646812701566e-06,
    "mesh_p95_error_percent": 1.3567502693656655e-15,
    "mesh_sampled_max_percent": 0.008187791571525698,
    "mesh_f1_0_1_percent": 100.0,
    "mesh_f1_0_5_percent": 100.0,
    "prepared_image_error_percent": 0.0,
    "prepared_image_changed_pixels_percent": 0.0,
    "prepared_images_pixel_identical": true,
    "prepared_reference_png_sha256": "aec8cb9c7a12d69c7743a365f64b6346217ffa9c433cf3b32dac192bb61f9cdd",
    "prepared_candidate_png_sha256": "aec8cb9c7a12d69c7743a365f64b6346217ffa9c433cf3b32dac192bb61f9cdd",
    "camera_yaw_error_degrees": 0.0,
    "camera_pitch_error_degrees": 0.0,
    "camera_field_of_view_error_degrees": 0.0,
    "camera_radius_error_percent": 0.0
  },
  {
    "object": "Kanex_MultiSync_Wireless_Keyboard",
    "identical_ply_bytes": true,
    "relative_latent_error_percent": 0.0,
    "foreground_color_error_percent": 0.0,
    "silhouette_overlap_percent": 100.0,
    "mesh_mean_error_percent": 4.350840886636189e-07,
    "mesh_p95_error_percent": 6.462191657360571e-16,
    "mesh_sampled_max_percent": 0.0018127244636245673,
    "mesh_f1_0_1_percent": 100.0,
    "mesh_f1_0_5_percent": 100.0,
    "prepared_image_error_percent": 0.0,
    "prepared_image_changed_pixels_percent": 0.0,
    "prepared_images_pixel_identical": true,
    "prepared_reference_png_sha256": "0e784b3fb7a222cd30bdee9270e7832c0d79cfc922b003cd9fb677ca2a48ec90",
    "prepared_candidate_png_sha256": "0e784b3fb7a222cd30bdee9270e7832c0d79cfc922b003cd9fb677ca2a48ec90",
    "camera_yaw_error_degrees": 0.0,
    "camera_pitch_error_degrees": 0.0,
    "camera_field_of_view_error_degrees": 0.0,
    "camera_radius_error_percent": 0.0
  },
  {
    "object": "Cole_Hardware_Mug_Classic_Blue",
    "identical_ply_bytes": true,
    "relative_latent_error_percent": 0.0,
    "foreground_color_error_percent": 0.0,
    "silhouette_overlap_percent": 100.0,
    "mesh_mean_error_percent": 1.4962001253256703e-06,
    "mesh_p95_error_percent": 1.869341080577122e-15,
    "mesh_sampled_max_percent": 0.0061353962229968115,
    "mesh_f1_0_1_percent": 100.0,
    "mesh_f1_0_5_percent": 100.0,
    "prepared_image_error_percent": 0.0,
    "prepared_image_changed_pixels_percent": 0.0,
    "prepared_images_pixel_identical": true,
    "prepared_reference_png_sha256": "64d7b4f054821989cce3505ba2f5e355f8050bf539181c68dbb5b3439a5c9b43",
    "prepared_candidate_png_sha256": "64d7b4f054821989cce3505ba2f5e355f8050bf539181c68dbb5b3439a5c9b43",
    "camera_yaw_error_degrees": 0.0,
    "camera_pitch_error_degrees": 0.0,
    "camera_field_of_view_error_degrees": 0.0,
    "camera_radius_error_percent": 0.0
  },
  {
    "object": "TURBOPROP_AIRPLANE_WITH_PILOT",
    "identical_ply_bytes": true,
    "relative_latent_error_percent": 0.0,
    "foreground_color_error_percent": 0.0,
    "silhouette_overlap_percent": 100.0,
    "mesh_mean_error_percent": 2.209268868768833e-06,
    "mesh_p95_error_percent": 1.1473623266866291e-15,
    "mesh_sampled_max_percent": 0.010958648047114137,
    "mesh_f1_0_1_percent": 100.0,
    "mesh_f1_0_5_percent": 100.0,
    "prepared_image_error_percent": 0.0,
    "prepared_image_changed_pixels_percent": 0.0,
    "prepared_images_pixel_identical": true,
    "prepared_reference_png_sha256": "20aa841ac85d02d940bb4ff067750db5a8fe9f7e1a3f7e43afd9a9e3dbc480c4",
    "prepared_candidate_png_sha256": "20aa841ac85d02d940bb4ff067750db5a8fe9f7e1a3f7e43afd9a9e3dbc480c4",
    "camera_yaw_error_degrees": 0.0,
    "camera_pitch_error_degrees": 0.0,
    "camera_field_of_view_error_degrees": 0.0,
    "camera_radius_error_percent": 0.0
  }
]

## Worst matched cases

| Object | Seed | Mean mesh error (%) | Tight-threshold mesh F1 (%) | Foreground color error (%) |
|---|---:|---:|---:|---:|
| Lego_Friends_Mia | 2026 | 1.7151 | 5.239 | 28.586 |
| DPC_Thinsulate_Isolate_Gloves_Brown | 2026 | 0.44033 | 18.628 | 11.519 |
| HeavyDuty_Flashlight | 0 | 0.30391 | 27.342 | 4.4951 |
| Perricone_MD_Cold_Plasma | 0 | 0.3037 | 20.838 | 4.06 |
| Unmellow_Yellow_Tieks_Neon_Patent_Leather_Ballet_Flats | 0 | 0.26924 | 26.289 | 5.9372 |
| Threshold_Porcelain_Pitcher_White | 0 | 0.20444 | 44.472 | 1.0183 |
| Razer_Taipan_White_Ambidextrous_Gaming_Mouse | 2026 | 0.20145 | 46.176 | 3.4404 |
| BAGEL_WITH_CHEESE | 2026 | 0.19188 | 54.332 | 1.6939 |
| Black_and_Decker_TR3500SD_2Slice_Toaster | 42 | 0.19085 | 59.617 | 1.4982 |
| Perricone_MD_Cold_Plasma | 2026 | 0.18412 | 37.138 | 2.3232 |
| Black_and_Decker_TR3500SD_2Slice_Toaster | 0 | 0.17478 | 67.342 | 1.2274 |
| Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Natural_Sparkle_Suede_w34KNQ41csH | 2026 | 0.17447 | 48.695 | 2.9495 |
