# Nsight 커널 프로파일

입력 1024토큰 · 출력 256토큰 · decode 255회.
시간은 프로파일 진단값이며 독립 지연시간 표본은 `../results/`에 있다.

| 구간 | Launch 수 | 커널 이름 수 | 커널 합계 ms | GPU 구간 ms |
|---|---:|---:|---:|---:|
| prefill | 4912 | 57 | 154.888 | 226.353 |
| unknown | 3 | 2 | 0.016 | 3908.531 |
| decode | 520965 | 46 | 3685.277 | 3908.510 |

모듈 귀속 미확인 커널은 unattributed로 보존한다. 아래 이름은 표시용으로 잘랐으며 전체 이름은 `kernels.csv`에 있다.

| 구간 | 커널 | Launch 수 | 합계 ms | 구간 커널 시간 비중 |
|---|---|---:|---:|---:|
| prefill | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<` | 573 | 16.828 | 10.86% |
| prefill | `void batch_trsm_left_kernel<float, (int)64, (int)4, (int)3, (bool)0, (bool)1, (bool)1>(cublasTrsmBatchParams2<T1>, T1, c` | 36 | 16.528 | 10.67% |
| prefill | `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` | 54 | 13.192 | 8.52% |
| prefill | `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn` | 42 | 12.120 | 7.83% |
| prefill | `volta_sgemm_64x64_tn` | 324 | 8.996 | 5.81% |
| unknown | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::Half, at::native::ArgMaxOps<float>, unsigned ` | 1 | 0.011 | 71.14% |
| unknown | `void at::native::<unnamed>::CatArrayBatchedCopy_alignedK_contig<at::native::<unnamed>::OpaqueType<(unsigned int)8>, unsi` | 2 | 0.005 | 28.86% |
| decode | `void gemv2T_kernel_val<int, int, __half, __half, __half, float, (int)128, (int)16, (int)4, (int)4, (bool)0, (bool)0, cub` | 12240 | 836.397 | 22.70% |
| decode | `std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, (bool)1, (bool)1, (bool)` | 255 | 574.253 | 15.58% |
| decode | `_decode_attention_kernel` | 1530 | 424.784 | 11.53% |
| decode | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, (bool)0, (bool)1, (bool` | 12240 | 333.216 | 9.04% |
| decode | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (i` | 75735 | 269.166 | 7.30% |

Memcpy·memset은 커널 합계에서 제외한다. Nsight Compute 표본은 별도 진단이며 전체 생성의 FLOPs·DRAM 트래픽으로 환산하지 않는다.
