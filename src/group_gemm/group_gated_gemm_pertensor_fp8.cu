// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "src/group_gemm/group_gemm.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace group_gemm {
namespace kernels {

// Reference fused gated GEMM kernel. Gate and up share the input traversal and
// activation/quantization happens before the result ever reaches global memory.
// This establishes the fused execution path; its math loop can subsequently be
// replaced by the WGMMA tiled implementation without changing the public API.
template <bool kUseBFloat16PrecisionMultiply, bool kUsePDL>
__global__ void group_gated_gemm_fp8_kernel(
    __nv_fp8_e4m3 *__restrict__ output, const __nv_fp8_e4m3 *__restrict__ input,
    const __nv_fp8_e4m3 *__restrict__ weight, const int *__restrict__ cu_seqlens,
    const float *__restrict__ gate_up_scale, const float *__restrict__ act_scale,
    int num_group, int m, int n, int k) {
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int valid_rows = cu_seqlens[num_group];
  if (row < m && row < valid_rows && col < n) {
    int left = 0;
    int right = num_group;
    while (left < right) {
      int mid = (left + right) / 2;
      if (cu_seqlens[mid + 1] <= row) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    int group = left;
    const auto *x = input + static_cast<int64_t>(row) * k;
    const auto *w = weight + static_cast<int64_t>(group) * (2 * n) * k;
    const auto *w_gate = w + static_cast<int64_t>(col) * k;
    const auto *w_up = w + static_cast<int64_t>(n + col) * k;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int ik = 0; ik < k; ++ik) {
      float xv = static_cast<float>(x[ik]);
      gate_acc = fmaf(xv, static_cast<float>(w_gate[ik]), gate_acc);
      up_acc = fmaf(xv, static_cast<float>(w_up[ik]), up_acc);
    }

    // The unfused GEMM stores BF16 before activation, so preserve that exact
    // rounding boundary in the fused epilogue.
    float scale = gate_up_scale[group];
    float gate = __bfloat162float(__float2bfloat16_rn(gate_acc * scale));
    float up = __bfloat162float(__float2bfloat16_rn(up_acc * scale));
    float value;
    if constexpr (kUseBFloat16PrecisionMultiply) {
      auto up_bf16 = __float2bfloat16_rn(up);
      value = __bfloat162float(__float2bfloat16_rn(silu(gate)) * up_bf16);
    } else {
      value = silu(gate) * up;
    }
    output[static_cast<int64_t>(row) * n + col] =
        __nv_fp8_e4m3(value * act_scale[0]);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

void group_gated_gemm_fp8_async(
    void *y_ptr, const void *x_ptr, const void *gate_up_weight_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *gate_up_scale_ptr, const void *act_scale_ptr, int num_group,
    int m, int n, int k, bool use_bf16_mul, bool use_pdl,
    cudaStream_t stream) {
  (void)seqlens_ptr;
  constexpr int kThreads = 256;
  dim3 block(kThreads);
  dim3 grid((n + kThreads - 1) / kThreads, m);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.stream = stream;
  cudaLaunchAttribute attribute[1];
  if (use_pdl) {
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
  }

#define LAUNCH_GATED(USE_BF16, USE_PDL)                                                \
  do {                                                                                 \
    auto kernel = kernels::group_gated_gemm_fp8_kernel<USE_BF16, USE_PDL>;             \
    cudaLaunchKernelEx(&config, kernel, static_cast<__nv_fp8_e4m3 *>(y_ptr),            \
                       static_cast<const __nv_fp8_e4m3 *>(x_ptr),                       \
                       static_cast<const __nv_fp8_e4m3 *>(gate_up_weight_ptr),          \
                       static_cast<const int *>(cu_seqlens_ptr),                        \
                       static_cast<const float *>(gate_up_scale_ptr),                   \
                       static_cast<const float *>(act_scale_ptr), num_group, m, n, k);  \
  } while (0)

  if (use_pdl) {
    if (use_bf16_mul) {
      LAUNCH_GATED(true, true);
    } else {
      LAUNCH_GATED(false, true);
    }
  } else if (use_bf16_mul) {
    LAUNCH_GATED(true, false);
  } else {
    LAUNCH_GATED(false, false);
  }
#undef LAUNCH_GATED
}

}  // namespace group_gemm
}  // namespace hpc
