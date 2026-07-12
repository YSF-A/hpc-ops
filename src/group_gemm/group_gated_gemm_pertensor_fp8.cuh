#include <cuda_bf16.h>

#include "cute/tensor.hpp"
#include "src/group_gemm/config.h"
#include "src/group_gemm/kernels.cuh"

namespace hpc {
namespace group_gemm {
namespace kernels {

template <typename Config, typename TmaA, typename TmaBGate, typename TmaBUp,
          bool kUseBFloat16PrecisionMultiply, bool kUsePDL>
__global__ void __launch_bounds__(384, 1) group_gated_gemm_fp8_kernel(
    const __grid_constant__ TmaBGate tma_b_gate, const __grid_constant__ TmaBUp tma_b_up,
    cute::TmaDescriptor *td_xy, const int *seqlens_ptr, const int *cu_seqlens_ptr,
    const float *gate_up_scale_ptr, const float *act_scale_ptr, int *tiles_ptr,
    cute::float_e4m3_t *output_ptr, int num_group, int m, int n, int k,
    cutlass::FastDivmod flat_divider) {
  using namespace cute;  // NOLINT
  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMma = typename Config::TiledMma;
  using SLayoutA = typename Config::SLayoutX;
  using SLayoutB = typename Config::SLayoutW;
  using SLayoutC = typename Config::SLayoutY;
  using TFused = std::conditional_t<kUseBFloat16PrecisionMultiply, Tout, float>;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;
  constexpr int kMmaThreads = size(TiledMma{});

  int idx = threadIdx.x;
  int iwarp = __shfl_sync(0xffffffff, idx / 32, 0);
  int elected = elect_one_sync();
  __shared__ uint64_t writable[kStage];
  __shared__ uint64_t readable[kStage];

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = reinterpret_cast<Tin *>(shm_data);
  auto *shm_gate = shm_a + cosize(SLayoutA{});
  auto *shm_up = shm_gate + cosize(SLayoutB{});
  auto *shm_out = reinterpret_cast<TFused *>(shm_up + cosize(SLayoutB{}));
  auto *shm_tiles = reinterpret_cast<int *>(shm_out + cosize(SLayoutC{}));

  TmaA tma_a;
  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sGate = make_tensor(make_smem_ptr(shm_gate), SLayoutB{});
  auto sUp = make_tensor(make_smem_ptr(shm_up), SLayoutB{});
  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gBGate = tma_b_gate.get_tma_tensor(make_shape(n, k, num_group));
  auto gBUp = tma_b_up.get_tma_tensor(make_shape(n, k, num_group));
  auto btma_a = tma_a.get_slice(0);
  auto btma_b_gate = tma_b_gate.get_slice(0);
  auto btma_b_up = tma_b_up.get_slice(0);
  auto tAg = btma_a.partition_S(gA);
  auto tAs = btma_a.partition_D(sA);
  auto tBGateG = btma_b_gate.partition_S(gBGate);
  auto tBUpG = btma_b_up.partition_S(gBUp);
  auto tBGateS = btma_b_gate.partition_D(sGate);
  auto tBUpS = btma_b_up.partition_D(sUp);

  if (idx < kStage) {
    initialize_barrier(readable[idx], 1);
    initialize_barrier(writable[idx], kMmaThreads);
  }
  for (int i = idx; i < num_group; i += blockDim.x) {
    shm_tiles[i] = tiles_ptr[i];
  }
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  __syncthreads();

  if (idx >= kMmaThreads) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= kMmaThreads;
    int load_warp = __shfl_sync(0xffffffff, idx / 32, 0);
    if (load_warp == 0 && elected) {
      constexpr int kTransactionBytes = sizeof(Tin) * (kTileM + 2 * kTileN) * kTileK;
      int phase = 1;
      int ismem_write = 0;
      int iblock = blockIdx.x;
      int igroup = 0;
      int sum_tile_m = 0;
      int itile_m, itile_n;
      int ntile_k = size<2>(tAg);
      while (true) {
        get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                              sum_tile_m, flat_divider);
        if (igroup < 0) break;
        iblock += gridDim.x;
        auto *td_x = td_xy + igroup * 2;
#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(writable[ismem_write], phase);
          copy(tma_a.with(td_x, readable[ismem_write]), tAg(_, itile_m, itile_k),
               tAs(_, 0, 0, ismem_write));
          copy(tma_b_gate.with(readable[ismem_write]),
               tBGateG(_, itile_n, itile_k, igroup), tBGateS(_, 0, 0, ismem_write));
          copy(tma_b_up.with(readable[ismem_write]),
               tBUpG(_, itile_n, itile_k, igroup), tBUpS(_, 0, 0, ismem_write));
          set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);
          if (++ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();
    int iwarpgroup = idx / 128;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tGateR = thr_mma.make_fragment_A(thr_mma.partition_A(sGate));
    auto tUpR = thr_mma.make_fragment_A(thr_mma.partition_A(sUp));
    auto tAR = thr_mma.make_fragment_B(thr_mma.partition_B(sA));
    auto dummy_c = make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                               make_shape(Int<kTileN>{}, Int<kTileM>{}),
                               make_stride(Int<kTileM>{}, Int<1>{}));
    auto tGateC = thr_mma.partition_fragment_C(dummy_c);
    auto tUpC = make_tensor_like(tGateC);
    int ismem_read = 0;
    int phase = 0;
    int iblock = blockIdx.x;
    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    while (true) {
      get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                            sum_tile_m, flat_divider);
      if (igroup < 0) break;
      iblock += gridDim.x;
      auto tGateD = make_tensor_like(tGateC);
      auto tUpD = make_tensor_like(tUpC);
      clear(tGateD);
      clear(tUpD);
      float scale = gate_up_scale_ptr[igroup];
      int ntile_k = size<2>(tAg);
#pragma unroll 1
      for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
        wait_barrier(readable[ismem_read], phase);
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        warpgroup_fence_operand(tGateC);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tAR); ++ik) {
          gemm(tiled_mma, tGateR(_, _, ik, ismem_read), tAR(_, _, ik, ismem_read),
               tGateC(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tGateC);

        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        warpgroup_fence_operand(tUpC);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tAR); ++ik) {
          gemm(tiled_mma, tUpR(_, _, ik, ismem_read), tAR(_, _, ik, ismem_read),
               tUpC(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tUpC);
        arrive_barrier(writable[ismem_read]);
#pragma unroll
        for (int i = 0; i < size(tGateC); ++i) {
          tGateD(i) += tGateC(i) * scale;
          tUpD(i) += tUpC(i) * scale;
        }
        if (++ismem_read == kStage) {
          ismem_read = 0;
          phase ^= 1;
        }
      }

      // Fuse the two accumulator fragments in registers. The explicit BF16
      // conversions preserve the rounding boundary of the unfused path, but
      // gate/up no longer make a round trip through shared memory.
      auto tOut = make_tensor_like<TFused>(tGateC);
#pragma unroll
      for (int i = 0; i < size(tGateC); ++i) {
        float gate = static_cast<float>(static_cast<Tout>(tGateD(i)));
        float up = static_cast<float>(static_cast<Tout>(tUpD(i)));
        if constexpr (kUseBFloat16PrecisionMultiply) {
          tOut(i) = __bfloat162float(__float2bfloat16_rn(silu(gate)) *
                                    __float2bfloat16_rn(up));
        } else {
          tOut(i) = silu(gate) * up;
        }
      }
      auto sOut = make_tensor(make_smem_ptr(shm_out), SLayoutC{});
      using STSMAtom = std::conditional_t<kTileM == 8, SM90_U16x4_STSM_T, SM90_U16x8_STSM_T>;
      using R2SCopyAtom = std::conditional_t<
          kUseBFloat16PrecisionMultiply, Copy_Atom<STSMAtom, Tout>,
          Copy_Atom<UniversalCopy<uint32_t>, float>>;
      auto tiled_copy = make_tiled_copy_C(R2SCopyAtom{}, tiled_mma);
      auto thr_copy = tiled_copy.get_slice(idx);
      auto tSOut = thr_copy.partition_D(sOut);
      syncwarpgroup(iwarpgroup);
      copy(tiled_copy, thr_copy.retile_S(tOut), tSOut);
      syncwarpgroup(iwarpgroup);

      constexpr int kWarpgroupTileN = kTileN / Config::kWarpgroupM;
      int group_rows = seqlens_ptr[igroup];
      for (int linear = idx % 128; linear < kWarpgroupTileN * kTileM; linear += 128) {
        int local_n = iwarpgroup * kWarpgroupTileN + linear / kTileM;
        int local_m = linear % kTileM;
        int row = cu_seqlens_ptr[igroup] + itile_m * kTileM + local_m;
        int col = itile_n * kTileN + local_n;
        if (local_m < group_rows - itile_m * kTileM && row < m && col < n) {
          output_ptr[static_cast<int64_t>(row) * n + col] =
              static_cast<cute::float_e4m3_t>(sOut(local_n, local_m) * act_scale_ptr[0]);
        }
      }
      syncwarpgroup(iwarpgroup);
    }
  }
  if constexpr (kUsePDL) cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace kernels

template <int kTileM, int kStage, bool kUseBFloat16PrecisionMultiply, bool kUsePDL>
void launch_group_gated_gemm_fp8(
    void *y_ptr, const void *x_ptr, const void *weight_ptr, const void *seqlens_ptr,
    const void *cu_seqlens_ptr, const void *gate_up_scale_ptr, const void *act_scale_ptr,
    void *tmas_ptr, void *tiles_ptr, int num_group, int m, int n, int k, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using Config = GroupGEMMFp8Config<Tin, Tout, kTileM, kTileN, kTileK, kStage, 2, 1, 128, 128, 64>;
  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto WGate = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(weight_ptr)),
                           make_shape(n, k, num_group), make_stride(k, Int<1>{}, 2 * n * k));
  auto WUp = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(weight_ptr) + n * k),
                         make_shape(n, k, num_group), make_stride(k, Int<1>{}, 2 * n * k));
  auto tma_b_gate = make_tma_copy(SM90_TMA_LOAD{}, WGate, take<0, 2>(typename Config::SLayoutW{}));
  auto tma_b_up = make_tma_copy(SM90_TMA_LOAD{}, WUp, take<0, 2>(typename Config::SLayoutW{}));
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, X, take<0, 2>(typename Config::SLayoutX{}));
  cutlass::FastDivmod flat_divider((n + kTileN - 1) / kTileN);
  int shm_size = (cosize(typename Config::SLayoutX{}) + 2 * cosize(typename Config::SLayoutW{})) *
                     sizeof(Tin) +
                 cosize(typename Config::SLayoutY{}) *
                     sizeof(std::conditional_t<kUseBFloat16PrecisionMultiply, Tout, float>) +
                 sizeof(int) * num_group;
  auto kernel = kernels::group_gated_gemm_fp8_kernel<
      Config, decltype(tma_a), decltype(tma_b_gate), decltype(tma_b_up),
      kUseBFloat16PrecisionMultiply, kUsePDL>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = get_sm_count();
  cfg.blockDim = 384;
  cfg.dynamicSmemBytes = shm_size;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  if constexpr (kUsePDL) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  cudaLaunchKernelEx(&cfg, kernel, tma_b_gate, tma_b_up,
                     static_cast<cute::TmaDescriptor *>(tmas_ptr),
                     static_cast<const int *>(seqlens_ptr),
                     static_cast<const int *>(cu_seqlens_ptr),
                     static_cast<const float *>(gate_up_scale_ptr),
                     static_cast<const float *>(act_scale_ptr), static_cast<int *>(tiles_ptr),
                     static_cast<Tin *>(y_ptr), num_group, m, n, k, flat_divider);
}

void group_gated_gemm_fp8_async(
    void *y_ptr, const void *x_ptr, const void *gate_up_weight_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *gate_up_scale_ptr, const void *act_scale_ptr, void *tmas_ptr,
    void *tiles_ptr, int num_group, int m, int n, int k,
    int num_seq_per_group_avg, bool use_bf16_mul, bool use_pdl, cudaStream_t stream) {
  use_pdl = true;
#define LAUNCH(TM, ST, BF, PDL)                                                        \
  launch_group_gated_gemm_fp8<TM, ST, BF, PDL>(                                       \
      y_ptr, x_ptr, gate_up_weight_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_scale_ptr,\
      act_scale_ptr, tmas_ptr, tiles_ptr, num_group, m, n, k, stream)
#define DISPATCH(BF, PDL)                       \
  do {                                          \
    if (num_seq_per_group_avg <= 8)             \
      LAUNCH(8, 5, BF, PDL);                    \
    else if (num_seq_per_group_avg <= 16)       \
      LAUNCH(16, 5, BF, PDL);                   \
    else if (num_seq_per_group_avg <= 32)       \
      LAUNCH(32, 4, BF, PDL);                   \
    else if (num_seq_per_group_avg <= 48)       \
      LAUNCH(48, 4, BF, PDL);                   \
    else if (num_seq_per_group_avg <= 64)       \
      LAUNCH(64, 3, BF, PDL);                   \
    else if (num_seq_per_group_avg <= 96)       \
      LAUNCH(48, 4, BF, PDL);                   \
    else if (num_seq_per_group_avg <= 128)      \
      LAUNCH(32, 4, BF, PDL);                   \
    else if (num_seq_per_group_avg <= 144)      \
      LAUNCH(48, 4, BF, PDL);                   \
    else                                        \
      LAUNCH(64, 3, BF, PDL);                   \
  } while (0)
  if (use_pdl) {
    if (use_bf16_mul) DISPATCH(true, true); else DISPATCH(false, true);
  } else {
    if (use_bf16_mul) DISPATCH(true, false); else DISPATCH(false, false);
  }
#undef DISPATCH
#undef LAUNCH
}

}  // namespace group_gemm
}  // namespace hpc
