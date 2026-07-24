# CUDA Run History

| Execution Time | Problem | Method | Platform | Status | Problem Size | Iterations | Average | Maximum | Minimum | Performance |
| --- | --- | --- | --- | :---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-07-20 03:00:16 CST | `01_Vector_Addition` | `legacy` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | N/A | N/A | 0.867 ms | 1.221 ms | 0.715 ms | N/A |
| 2026-07-20 10:53:12 CST | `01_Vector_Addition` | `legacy` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | N/A | N/A | 0.796 ms | 1.005 ms | 0.750 ms | N/A |
| 2026-07-21 17:14:38 CST | `01_Vector_Addition` | `legacy` | NVIDIA GeForce RTX 4090 | **PASS** | N/A | N/A | 0.328 ms | 0.331 ms | 0.326 ms | N/A |
| 2026-07-25 03:46:00 CST | `02_Matrix_Multiplication` | `v0_naive` | N/A | **FAIL** | N/A | N/A | N/A | N/A | N/A | N/A |
| 2026-07-25 03:46:26 CST | `02_Matrix_Multiplication` | `v1_shared_memory` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 63.839 ms | 65.444 ms | 60.773 ms | 6458.683 GFLOPS |
| 2026-07-25 03:46:46 CST | `02_Matrix_Multiplication` | `v0_naive` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 85.611 ms | 87.429 ms | 83.138 ms | 4816.171 GFLOPS |
| 2026-07-25 03:46:58 CST | `02_Matrix_Multiplication` | `v1_1d_thread_tiling` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 19.518 ms | 19.529 ms | 19.509 ms | 21124.798 GFLOPS |
| 2026-07-25 03:47:10 CST | `02_Matrix_Multiplication` | `v1_2d_thread_tiling` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 12.655 ms | 12.823 ms | 12.474 ms | 32580.496 GFLOPS |
| 2026-07-25 03:47:21 CST | `02_Matrix_Multiplication` | `v2_vectorized` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.800 ms | 9.108 ms | 8.644 ms | 46854.645 GFLOPS |
| 2026-07-25 03:49:28 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.739 ms | 8.742 ms | 8.737 ms | 47182.464 GFLOPS |
| Recovered from latest output | `02_Matrix_Multiplication` | `default` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 5.293 ms | 5.295 ms | 5.288 ms | 77902.088 GFLOPS |
| 2026-07-25 04:00:14 CST | `01_Vector_Addition` | `default` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | N=25000000 | 20 | 0.929 ms | 1.708 ms | 0.727 ms | 322.986 GB/s |
| 2026-07-25 04:01:06 CST | `02_Matrix_Multiplication` | `v0_naive` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 349.854 ms | 415.544 ms | 307.009 ms | 1178.539 GFLOPS |
| 2026-07-25 04:01:20 CST | `02_Matrix_Multiplication` | `v1_shared_memory` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 305.883 ms | 425.565 ms | 239.407 ms | 1347.956 GFLOPS |
| 2026-07-25 04:01:32 CST | `02_Matrix_Multiplication` | `v1_1d_thread_tiling` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 61.112 ms | 62.660 ms | 59.301 ms | 6746.892 GFLOPS |
| 2026-07-25 04:01:44 CST | `02_Matrix_Multiplication` | `v1_2d_thread_tiling` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 43.193 ms | 45.644 ms | 41.566 ms | 9545.823 GFLOPS |
| 2026-07-25 04:01:57 CST | `02_Matrix_Multiplication` | `v2_vectorized` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **FAIL** | N/A | N/A | N/A | N/A | N/A | N/A |
| 2026-07-25 04:02:03 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 33.830 ms | 35.849 ms | 32.570 ms | 12187.893 GFLOPS |
| 2026-07-25 04:02:15 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 19.633 ms | 21.787 ms | 16.761 ms | 21000.688 GFLOPS |
| 2026-07-25 04:02:38 CST | `02_Matrix_Multiplication` | `v2_vectorized` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 32.614 ms | 33.523 ms | 31.471 ms | 12642.353 GFLOPS |
| 2026-07-25 04:03:01 CST | `01_Vector_Addition` | `default` | NVIDIA GeForce RTX 4090 | **PASS** | N=25000000 | 20 | 0.379 ms | 1.332 ms | 0.327 ms | 792.152 GB/s |
| 2026-07-25 04:03:53 CST | `02_Matrix_Multiplication` | `default` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 5.289 ms | 5.293 ms | 5.285 ms | 77963.055 GFLOPS |
| 2026-07-25 04:08:41 CST | `02_Matrix_Multiplication` | `v0_naive` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 83.949 ms | 85.607 ms | 83.063 ms | 4911.543 GFLOPS |
| 2026-07-25 04:08:55 CST | `02_Matrix_Multiplication` | `v1_shared_memory` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 61.351 ms | 65.267 ms | 59.362 ms | 6720.632 GFLOPS |
| 2026-07-25 04:09:11 CST | `02_Matrix_Multiplication` | `v1_1d_thread_tiling` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 19.508 ms | 19.514 ms | 19.498 ms | 21135.588 GFLOPS |
| 2026-07-25 04:09:23 CST | `02_Matrix_Multiplication` | `v1_2d_thread_tiling` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 12.540 ms | 12.771 ms | 12.421 ms | 32880.130 GFLOPS |
| 2026-07-25 04:09:35 CST | `02_Matrix_Multiplication` | `v2_vectorized` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.642 ms | 8.643 ms | 8.641 ms | 47711.609 GFLOPS |
| 2026-07-25 04:09:48 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.773 ms | 8.898 ms | 8.709 ms | 46996.203 GFLOPS |
| 2026-07-25 04:10:00 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 4090 | **FAIL** | M=8192, N=6144, K=4096 | 3 | 5.288 ms | 5.291 ms | 5.284 ms | 77972.644 GFLOPS |
| 2026-07-25 04:10:54 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.739 ms | 8.742 ms | 8.737 ms | 47180.626 GFLOPS |
| 2026-07-25 04:11:10 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.285 ms | 9.288 ms | 9.284 ms | 44405.380 GFLOPS |
| 2026-07-25 04:11:44 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.009 ms | 9.027 ms | 8.999 ms | 45769.321 GFLOPS |
| 2026-07-25 04:12:31 CST | `02_Matrix_Multiplication` | `v0_naive` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 348.822 ms | 408.415 ms | 303.296 ms | 1182.026 GFLOPS |
| 2026-07-25 04:12:44 CST | `02_Matrix_Multiplication` | `v1_shared_memory` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 242.778 ms | 296.659 ms | 215.807 ms | 1698.330 GFLOPS |
| 2026-07-25 04:12:56 CST | `02_Matrix_Multiplication` | `v1_1d_thread_tiling` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 60.158 ms | 60.709 ms | 59.850 ms | 6853.909 GFLOPS |
| 2026-07-25 04:13:09 CST | `02_Matrix_Multiplication` | `v1_2d_thread_tiling` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 45.858 ms | 48.239 ms | 44.093 ms | 8991.259 GFLOPS |
| 2026-07-25 04:13:20 CST | `02_Matrix_Multiplication` | `v2_vectorized` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 33.253 ms | 34.445 ms | 31.828 ms | 12399.341 GFLOPS |
| 2026-07-25 04:13:33 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 36.399 ms | 38.037 ms | 34.970 ms | 11327.579 GFLOPS |
| 2026-07-25 04:13:45 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 28.646 ms | 29.744 ms | 27.221 ms | 14393.392 GFLOPS |
| 2026-07-25 04:14:46 CST | `02_Matrix_Multiplication` | `v5_adaptive` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 27.860 ms | 29.722 ms | 26.181 ms | 14799.668 GFLOPS |
| 2026-07-25 04:15:06 CST | `02_Matrix_Multiplication` | `v5_adaptive` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 10.044 ms | 10.960 ms | 9.583 ms | 41049.102 GFLOPS |
| 2026-07-25 04:15:38 CST | `02_Matrix_Multiplication` | `v5_adaptive` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.738 ms | 8.741 ms | 8.733 ms | 47188.397 GFLOPS |
| 2026-07-25 04:16:03 CST | `02_Matrix_Multiplication` | `v5_adaptive` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 28.646 ms | 29.861 ms | 27.397 ms | 14393.686 GFLOPS |
| 2026-07-25 04:17:52 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.819 ms | 8.979 ms | 8.736 ms | 46751.731 GFLOPS |
| 2026-07-25 04:18:06 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.764 ms | 8.859 ms | 8.715 ms | 47047.738 GFLOPS |
| 2026-07-25 04:18:20 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.755 ms | 8.791 ms | 8.735 ms | 47093.594 GFLOPS |
| 2026-07-25 04:18:32 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.037 ms | 9.073 ms | 9.015 ms | 45627.822 GFLOPS |
| 2026-07-25 04:18:47 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.238 ms | 9.679 ms | 9.016 ms | 44631.795 GFLOPS |
| 2026-07-25 04:19:01 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.454 ms | 10.093 ms | 9.017 ms | 43611.750 GFLOPS |
| 2026-07-25 04:19:34 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 35.975 ms | 39.961 ms | 32.919 ms | 11461.105 GFLOPS |
| 2026-07-25 04:19:46 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 35.262 ms | 36.849 ms | 33.533 ms | 11692.873 GFLOPS |
| 2026-07-25 04:19:57 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 33.903 ms | 37.208 ms | 30.154 ms | 12161.818 GFLOPS |
| 2026-07-25 04:20:09 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 28.361 ms | 28.769 ms | 27.734 ms | 14538.261 GFLOPS |
| 2026-07-25 04:20:21 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 29.496 ms | 29.891 ms | 28.998 ms | 13978.687 GFLOPS |
| 2026-07-25 04:20:34 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 3 | 29.029 ms | 29.768 ms | 28.421 ms | 14203.795 GFLOPS |
| 2026-07-25 04:23:56 CST | `02_Matrix_Multiplication` | `default` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.740 ms | 8.742 ms | 8.738 ms | 47178.031 GFLOPS |
| 2026-07-25 04:37:47 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.178 ms | 9.442 ms | 9.045 ms | 44926.801 GFLOPS |
| 2026-07-25 04:37:59 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.044 ms | 9.046 ms | 9.043 ms | 45590.745 GFLOPS |
| 2026-07-25 04:38:10 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.044 ms | 9.048 ms | 9.039 ms | 45590.904 GFLOPS |
| 2026-07-25 04:38:56 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.062 ms | 9.064 ms | 9.060 ms | 45498.177 GFLOPS |
| 2026-07-25 04:39:56 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.060 ms | 9.062 ms | 9.056 ms | 45510.979 GFLOPS |
| 2026-07-25 04:40:09 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 9.038 ms | 9.041 ms | 9.035 ms | 45619.527 GFLOPS |
| 2026-07-25 04:42:11 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 3 | 8.944 ms | 9.351 ms | 8.737 ms | 46098.288 GFLOPS |
| 2026-07-25 04:57:08 CST | `02_Matrix_Multiplication` | `v2_vectorized` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 11.526 ms | 11.717 ms | 11.359 ms | 35773.893 GFLOPS |
| 2026-07-25 04:57:27 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 12.214 ms | 12.783 ms | 11.304 ms | 33758.222 GFLOPS |
| 2026-07-25 04:57:52 CST | `02_Matrix_Multiplication` | `v2_vectorized` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 37.630 ms | 40.608 ms | 33.509 ms | 10957.192 GFLOPS |
| 2026-07-25 04:58:06 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 39.856 ms | 41.806 ms | 38.156 ms | 10345.265 GFLOPS |
| 2026-07-25 04:59:23 CST | `02_Matrix_Multiplication` | `v2_vectorized_bk16` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 11.035 ms | 11.805 ms | 10.414 ms | 37364.875 GFLOPS |
| 2026-07-25 04:59:43 CST | `02_Matrix_Multiplication` | `v2_vectorized_bk16` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 38.951 ms | 41.201 ms | 36.741 ms | 10585.661 GFLOPS |
| 2026-07-25 05:00:09 CST | `02_Matrix_Multiplication` | `v0_naive` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 83.107 ms | 83.530 ms | 82.699 ms | 4961.253 GFLOPS |
| 2026-07-25 05:00:21 CST | `02_Matrix_Multiplication` | `v1_shared_memory` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 59.324 ms | 59.929 ms | 58.925 ms | 6950.270 GFLOPS |
| 2026-07-25 05:00:33 CST | `02_Matrix_Multiplication` | `v1_1d_thread_tiling` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 19.505 ms | 19.528 ms | 19.444 ms | 21138.901 GFLOPS |
| 2026-07-25 05:00:45 CST | `02_Matrix_Multiplication` | `v1_2d_thread_tiling` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 13.354 ms | 13.925 ms | 12.488 ms | 30875.503 GFLOPS |
| 2026-07-25 05:00:57 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 10.295 ms | 10.640 ms | 10.012 ms | 40049.811 GFLOPS |
| 2026-07-25 05:01:16 CST | `02_Matrix_Multiplication` | `v0_naive` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 384.666 ms | 445.092 ms | 342.381 ms | 1071.882 GFLOPS |
| 2026-07-25 05:01:34 CST | `02_Matrix_Multiplication` | `v1_shared_memory` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 242.443 ms | 290.510 ms | 217.389 ms | 1700.675 GFLOPS |
| 2026-07-25 05:01:50 CST | `02_Matrix_Multiplication` | `v1_1d_thread_tiling` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 78.838 ms | 100.539 ms | 65.303 ms | 5229.916 GFLOPS |
| 2026-07-25 05:02:02 CST | `02_Matrix_Multiplication` | `v1_2d_thread_tiling` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 53.735 ms | 70.620 ms | 50.348 ms | 7673.130 GFLOPS |
| 2026-07-25 05:02:14 CST | `02_Matrix_Multiplication` | `v4_large_tile` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 32.258 ms | 34.764 ms | 29.824 ms | 12781.838 GFLOPS |
| 2026-07-25 05:05:25 CST | `02_Matrix_Multiplication` | `default` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 10.284 ms | 10.721 ms | 9.941 ms | 40094.689 GFLOPS |
| 2026-07-25 05:05:57 CST | `02_Matrix_Multiplication` | `default` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 32.567 ms | 34.811 ms | 30.380 ms | 12660.545 GFLOPS |
| 2026-07-25 05:13:57 CST | `02_Matrix_Multiplication` | `v2_vectorized_large_tile` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 12.730 ms | 13.177 ms | 12.409 ms | 32389.696 GFLOPS |
| 2026-07-25 05:14:22 CST | `02_Matrix_Multiplication` | `v2_vectorized_large_tile` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 44.230 ms | 47.437 ms | 42.647 ms | 9322.085 GFLOPS |
| 2026-07-25 05:25:32 CST | `02_Matrix_Multiplication` | `v0_naive` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 83.174 ms | 83.516 ms | 82.662 ms | 4957.308 GFLOPS |
| 2026-07-25 05:26:55 CST | `02_Matrix_Multiplication` | `v0_naive` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 83.201 ms | 83.467 ms | 82.706 ms | 4955.654 GFLOPS |
| 2026-07-25 05:27:28 CST | `02_Matrix_Multiplication` | `v1_shared_memory` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 59.247 ms | 60.184 ms | 58.716 ms | 6959.313 GFLOPS |
| 2026-07-25 05:28:03 CST | `02_Matrix_Multiplication` | `v1_1d_thread_tiling` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 19.415 ms | 19.487 ms | 19.372 ms | 21237.378 GFLOPS |
| 2026-07-25 05:28:37 CST | `02_Matrix_Multiplication` | `v1_2d_thread_tiling` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 13.560 ms | 13.786 ms | 13.360 ms | 30406.969 GFLOPS |
| 2026-07-25 05:29:09 CST | `02_Matrix_Multiplication` | `v2_vectorized` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 11.412 ms | 11.631 ms | 11.183 ms | 36131.138 GFLOPS |
| 2026-07-25 05:29:44 CST | `02_Matrix_Multiplication` | `v2_vectorized_bk16` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 11.004 ms | 11.589 ms | 10.156 ms | 37470.392 GFLOPS |
| 2026-07-25 05:30:16 CST | `02_Matrix_Multiplication` | `v2_vectorized_large_tile` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 12.735 ms | 13.171 ms | 12.387 ms | 32376.516 GFLOPS |
| 2026-07-25 05:30:51 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 10.737 ms | 11.663 ms | 8.714 ms | 38399.761 GFLOPS |
| 2026-07-25 05:31:26 CST | `02_Matrix_Multiplication` | `v3_large_tile` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 10.317 ms | 10.573 ms | 10.002 ms | 39963.267 GFLOPS |
| 2026-07-25 05:32:03 CST | `02_Matrix_Multiplication` | `default` | NVIDIA GeForce RTX 4090 | **PASS** | M=8192, N=6144, K=4096 | 10 | 10.399 ms | 10.715 ms | 10.008 ms | 39651.122 GFLOPS |
| 2026-07-25 05:32:54 CST | `02_Matrix_Multiplication` | `v0_naive` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 370.411 ms | 440.430 ms | 333.271 ms | 1113.135 GFLOPS |
| 2026-07-25 05:33:15 CST | `02_Matrix_Multiplication` | `v1_shared_memory` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 253.695 ms | 306.542 ms | 223.976 ms | 1625.249 GFLOPS |
| 2026-07-25 05:33:36 CST | `02_Matrix_Multiplication` | `v1_1d_thread_tiling` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 81.154 ms | 105.337 ms | 67.253 ms | 5080.654 GFLOPS |
| 2026-07-25 05:33:54 CST | `02_Matrix_Multiplication` | `v1_2d_thread_tiling` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 54.954 ms | 74.808 ms | 46.046 ms | 7502.970 GFLOPS |
| 2026-07-25 05:34:11 CST | `02_Matrix_Multiplication` | `v2_vectorized` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 39.486 ms | 45.050 ms | 36.339 ms | 10442.033 GFLOPS |
| 2026-07-25 05:34:28 CST | `02_Matrix_Multiplication` | `v2_vectorized_bk16` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 40.241 ms | 44.580 ms | 37.494 ms | 10246.190 GFLOPS |
| 2026-07-25 05:34:45 CST | `02_Matrix_Multiplication` | `v2_vectorized_large_tile` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 45.082 ms | 47.103 ms | 42.618 ms | 9145.932 GFLOPS |
| 2026-07-25 05:35:01 CST | `02_Matrix_Multiplication` | `v3_double_buffered` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 40.441 ms | 43.165 ms | 38.539 ms | 10195.625 GFLOPS |
| 2026-07-25 05:35:17 CST | `02_Matrix_Multiplication` | `v3_large_tile` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 33.051 ms | 34.255 ms | 31.191 ms | 12475.352 GFLOPS |
| 2026-07-25 05:35:34 CST | `02_Matrix_Multiplication` | `default` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | M=8192, N=6144, K=4096 | 10 | 32.366 ms | 35.650 ms | 29.716 ms | 12739.032 GFLOPS |
| 2026-07-25 05:43:45 CST | `01_Vector_Addition` | `default` | NVIDIA GeForce RTX 4090 | **PASS** | N=25000000 | 20 | 0.324 ms | 0.325 ms | 0.321 ms | 927.272 GB/s |
| 2026-07-25 05:45:55 CST | `01_Vector_Addition` | `default` | NVIDIA GeForce RTX 5070 Ti Laptop GPU | **PASS** | N=25000000 | 20 | 0.826 ms | 1.325 ms | 0.748 ms | 363.144 GB/s |
