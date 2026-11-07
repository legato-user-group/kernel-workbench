# Nsight 커널 프로파일

입력 1024토큰 · 출력 256토큰 · decode 255회.
시간은 프로파일 진단값이며 독립 지연시간 표본은 `../results/`에 있다.

| 구간 | Launch 수 | 커널 이름 수 | 커널 합계 ms | GPU 구간 ms |
|---|---:|---:|---:|---:|
| prefill | 4912 | 57 | 158.275 | 249.422 |
| unknown | 3 | 2 | 0.017 | 4058.278 |
| decode | 520965 | 46 | 3787.018 | 4058.255 |

모듈 귀속 미확인 커널은 unattributed로 보존한다. 아래 이름은 표시용으로 잘랐으며 전체 이름은 `kernels.csv`에 있다.

| 구간 | 커널 | Launch 수 | 합계 ms | 구간 커널 시간 비중 |
|---|---|---:|---:|---:|
| prefill | `void batch_trsm_left_kernel<float, (int)64, (int)4, (int)3, (bool)0, (bool)1, (bool)1>(cublasTrsmBatchParams2<T1>, T1, c` | 36 | 17.043 | 10.77% |
| prefill | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<` | 573 | 16.879 | 10.66% |
| prefill | `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` | 54 | 13.667 | 8.64% |
| prefill | `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn` | 42 | 12.510 | 7.90% |
| prefill | `volta_sgemm_64x64_tn` | 324 | 9.115 | 5.76% |
| unknown | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::Half, at::native::ArgMaxOps<float>, unsigned ` | 1 | 0.012 | 70.45% |
| unknown | `void at::native::<unnamed>::CatArrayBatchedCopy_alignedK_contig<at::native::<unnamed>::OpaqueType<(unsigned int)8>, unsi` | 2 | 0.005 | 29.55% |
| decode | `void gemv2T_kernel_val<int, int, __half, __half, __half, float, (int)128, (int)16, (int)4, (int)4, (bool)0, (bool)0, cub` | 12240 | 839.811 | 22.18% |
| decode | `std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, (bool)1, (bool)1, (bool)` | 255 | 575.260 | 15.19% |
| decode | `_decode_attention_kernel` | 1530 | 425.287 | 11.23% |
| decode | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, (bool)0, (bool)1, (bool` | 12240 | 338.061 | 8.93% |
| decode | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (i` | 75735 | 287.420 | 7.59% |

Memcpy·memset은 커널 합계에서 제외한다. Nsight Compute 표본은 별도 진단이며 전체 생성의 FLOPs·DRAM 트래픽으로 환산하지 않는다.
