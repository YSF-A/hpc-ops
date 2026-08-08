#include <cuda_bf16.h>
#include <stdint.h>

#include "cute/tensor.hpp"
#include "src/group_gemm/config.h"
#include "src/group_gemm/kernels.cuh"

namespace hpc {
namespace group_gemm {
namespace kernels {

template <typename Config, typename TmaA, typename TmaBGate, typename TmaBUp, typename TmaD,
          int kTaskLoopPolicy, bool kUseBFloat16PrecisionMultiply, bool kUsePDL>
__global__ void __launch_bounds__(384, 1) group_gated_gemm_fp8_kernel(
    const __grid_constant__ TmaBGate tma_b_gate, const __grid_constant__ TmaBUp tma_b_up,
    const __grid_constant__ TmaD tma_d,
    cute::TmaDescriptor *td_xy, const int *seqlens_ptr, const int *cu_seqlens_ptr,
    const float *gate_up_scale_ptr, const float *act_scale_ptr, int *tiles_ptr,
    int *cu_tiles_ptr, int4 *task_map_ptr, typename Config::Tin *output_ptr, int num_group,
    int m, int n, int k,
    cutlass::FastDivmod flat_divider) {
  using namespace cute;  // NOLINT
  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMma = typename Config::TiledMma;
  using SLayoutA = typename Config::SLayoutX;
  using SLayoutB = typename Config::SLayoutW;
  using SLayoutDAtom = decltype(slayout_selector<64, Tin, false>());
  using SLayoutD =
      decltype(tile_to_shape(SLayoutDAtom{}, cute::make_shape(cute::Int<Config::kTileN>{},
                                                              cute::Int<Config::kTileM>{})));
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;
  constexpr int kMmaThreads = size(TiledMma{});

  int idx = threadIdx.x;
  int iwarp = __shfl_sync(0xffffffff, idx / 32, 0);
  int ilane = idx % 32;
  int elected = elect_one_sync();
  __shared__ uint64_t writable[kStage];
  __shared__ uint64_t readable[kStage];

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = reinterpret_cast<Tin *>(shm_data);
  auto *shm_gate = shm_a + cosize(SLayoutA{});
  auto *shm_up = shm_gate + cosize(SLayoutB{});
  auto *shm_out = reinterpret_cast<Tin *>(shm_up + cosize(SLayoutB{}));
  using Ttask = std::conditional_t<kTaskLoopPolicy == 0, int4, int>;
  auto *shm_tiles = reinterpret_cast<Ttask *>(shm_out + cosize(SLayoutD{}));

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
  int num_tile_n = size<1>(tBGateG);

  if (idx < kStage) {
    initialize_barrier(readable[idx], 1);
    initialize_barrier(writable[idx], kMmaThreads);
  }
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  int total_m = 0;
  int actual_tiles = 0;
  if constexpr (kTaskLoopPolicy == 0) {
    int warp_count = blockDim.x / 32;
    int iwarp_block = blockIdx.x + gridDim.x * iwarp;
    actual_tiles = cu_tiles_ptr[num_group] * num_tile_n + gridDim.x;
    int iwave = 0;
    while (iwarp_block < actual_tiles) {
      int4 task = task_map_ptr[iwarp_block];
      if (ilane == 0) {
        shm_tiles[iwave * warp_count + iwarp] = task;
      }
      int igroup = task.z;
      if (igroup < 0) {
        break;
      }
      tma_descriptor_fence_acquire(td_xy + igroup * 2);
      tma_descriptor_fence_acquire(td_xy + igroup * 2 + 1);
      iwarp_block += gridDim.x * warp_count;
      iwave++;
    }
  } else if constexpr (kTaskLoopPolicy == 1) {
    for (int i = idx; i < num_group; i += blockDim.x) {
      shm_tiles[i] = tiles_ptr[i];
    }
  } else if constexpr (kTaskLoopPolicy == 2) {
    total_m = cu_tiles_ptr[num_group];
    for (int i = idx; i < num_group + 1; i += blockDim.x) {
      shm_tiles[i] = cu_tiles_ptr[i];
    }
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
      int4 task;
      int iwave = 0;
      int ntile_k = size<2>(tAg);
      while (true) {
        if constexpr (kTaskLoopPolicy == 0) {
          task = shm_tiles[iwave];
          itile_m = task.x;
          itile_n = task.y;
          igroup = task.z;
          if (igroup < 0) break;
          iwave++;
        } else if constexpr (kTaskLoopPolicy == 1) {
          get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                sum_tile_m, flat_divider);
          if (igroup < 0) break;
        } else {
          get_next_tile_vert(shm_tiles, iblock, num_group, igroup, itile_m, itile_n, total_m);
          if (itile_n >= num_tile_n) break;
        }
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
    int4 task;
    int iwave = 0;

    while (true) {
      if constexpr (kTaskLoopPolicy == 0) {
        task = shm_tiles[iwave];
        itile_m = task.x;
        itile_n = task.y;
        igroup = task.z;
        if (igroup < 0) break;
        iwave++;
      } else if constexpr (kTaskLoopPolicy == 1) {
        get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                              sum_tile_m, flat_divider);
        if (igroup < 0) break;
      } else {
        get_next_tile_vert(shm_tiles, iblock, num_group, igroup, itile_m, itile_n, total_m);
        if (itile_n >= num_tile_n) break;
      }
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
      auto tOut = make_tensor_like<Tin>(tGateC);
#pragma unroll
      for (int i = 0; i < size(tGateC); ++i) {
        float gate = static_cast<float>(static_cast<Tout>(tGateD(i)));
        float up = static_cast<float>(static_cast<Tout>(tUpD(i)));
        float fused;
        if constexpr (kUseBFloat16PrecisionMultiply) {
          fused = __bfloat162float(__float2bfloat16_rn(silu(gate)) *
                                   __float2bfloat16_rn(up));
        } else {
          fused = silu(gate) * up;
        }
        tOut(i) = static_cast<Tin>(fused * act_scale_ptr[0]);
      }
      auto sOut = make_tensor(make_smem_ptr(shm_out), SLayoutD{});
      using R2SCopyAtom = Copy_Atom<UniversalCopy<uint8_t>, Tin>;
      auto tiled_copy = make_tiled_copy_C(R2SCopyAtom{}, tiled_mma);
      auto thr_copy = tiled_copy.get_slice(idx);
      auto tSOut = thr_copy.partition_D(sOut);
      tma_store_wait<0>();
      syncwarpgroup(iwarpgroup);
      copy(tiled_copy, thr_copy.retile_S(tOut), tSOut);
      syncwarpgroup(iwarpgroup);
      int group_rows = seqlens_ptr[igroup];
      int row_start = cu_seqlens_ptr[igroup] + itile_m * Config::kTileM;
      // Keep the grouped writeback scalar while validating the dynamic output
      // descriptor path. Input and weight transfers remain TMA based.
      constexpr int kWarpgroupTileN = Config::kTileN / Config::kWarpgroupM;
#pragma unroll
      for (int linear = idx % 128; linear < kWarpgroupTileN * Config::kTileM;
           linear += 128) {
        int local_n = iwarpgroup * kWarpgroupTileN + linear / Config::kTileM;
        int local_m = linear % Config::kTileM;
        int row = row_start + local_m;
        int col = itile_n * Config::kTileN + local_n;
        if (local_m < group_rows - itile_m * Config::kTileM && row < m && col < n) {
          output_ptr[static_cast<int64_t>(row) * n + col] = sOut(local_n, local_m);
        }
      }
    }
  }
  if constexpr (kUsePDL) cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace kernels

namespace kernels {

template <typename Tout, typename TmaD, bool kUsePDL>
__global__ void update_grouped_gated_output_tma(cute::TmaDescriptor td_y,
                                                cute::TmaDescriptor *td_xy,
                                                const Tout *y_ptr, const int *seqlens_ptr,
                                                const int *cu_seqlens_ptr, int num_group,
                                                int m, int n) {
  using namespace cute;  // NOLINT
  int idx = threadIdx.x;
  int igroup = blockIdx.x;

  constexpr bool kEnableOutputTmaStore = false;
  if constexpr (kEnableOutputTmaStore && kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if (igroup < num_group) {
    __shared__ cute::TmaDescriptor smem_tma_desc;
    int num_seq = seqlens_ptr[igroup];
    uint64_t cu_seqlen = cu_seqlens_ptr[igroup];
    auto *y_ibatch_ptr = y_ptr + cu_seqlen * n;

    if (idx == 0) {
      smem_tma_desc = td_y;
    }
    __syncwarp();

    if (idx == 0) {
      auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(n, num_seq),
                            make_stride(Int<1>{}, n));
      update_tma_gtensor<TmaD>(smem_tma_desc, gY);
    }

    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    tma_descriptor_cp_fence_release(td_xy + igroup * 2 + 1, smem_tma_desc);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <bool kUsePDL>
__global__ void build_gated_task_map_kernel(int4 *task_map_ptr, const int *cu_tiles_ptr,
                                            const int *tiles_ptr, int num_group,
                                            int num_tile_n, int task_map_len) {
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  int igroup = blockIdx.x;
  if (igroup < num_group) {
    int cu_tile_m = cu_tiles_ptr[igroup];
    int num_tile_m = tiles_ptr[igroup];
    int cu_tiles = cu_tile_m * num_tile_n;
    int total = num_tile_m * num_tile_n;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
      int itile_m = i / num_tile_n;
      int itile_n = i - itile_m * num_tile_n;
      int4 task;
      task.x = itile_m;
      task.y = itile_n;
      task.z = igroup;
      task.w = 0;
      task_map_ptr[cu_tiles + i] = task;
    }
  } else {
    int tail_id = blockIdx.x - num_group;
    int tail_blocks = gridDim.x - num_group;
    int used = cu_tiles_ptr[num_group] * num_tile_n;
    int4 sentinel;
    sentinel.x = 0;
    sentinel.y = 0;
    sentinel.z = -1;
    sentinel.w = 0;
    int stride = tail_blocks * blockDim.x;
    for (int i = used + tail_id * blockDim.x + threadIdx.x; i < task_map_len; i += stride) {
      task_map_ptr[i] = sentinel;
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

template <int kTileM, int kStage, int kTaskLoopPolicy, bool kUseBFloat16PrecisionMultiply,
          bool kUsePDL>
void launch_group_gated_gemm_fp8(
    void *y_ptr, const void *x_ptr, const void *weight_ptr, const void *seqlens_ptr,
    const void *cu_seqlens_ptr, const void *gate_up_scale_ptr, const void *act_scale_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, void *task_map_ptr, int num_waves,
    int num_group, int m, int n, int k, cudaStream_t stream) {
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
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tin *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));
  using SLayoutDAtom = decltype(slayout_selector<64, Tin, false>());
  using SLayoutD = decltype(tile_to_shape(SLayoutDAtom{}, make_shape(Int<kTileN>{}, Int<kTileM>{})));
  using CopyBoxD = decltype(tile_to_shape(
      SLayoutDAtom{}, make_shape(Int<kTileN / Config::kWarpgroupM>{}, Int<kTileM>{})));
  auto tma_b_gate = make_tma_copy(SM90_TMA_LOAD{}, WGate, take<0, 2>(typename Config::SLayoutW{}));
  auto tma_b_up = make_tma_copy(SM90_TMA_LOAD{}, WUp, take<0, 2>(typename Config::SLayoutW{}));
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, X, take<0, 2>(typename Config::SLayoutX{}));
  auto tma_d = make_tma_copy(SM90_TMA_STORE{}, Y, CopyBoxD{});
  int num_tile_n = (n + kTileN - 1) / kTileN;
  cutlass::FastDivmod flat_divider(num_tile_n);
  if constexpr (kTaskLoopPolicy == 0) {
    int task_map_len = num_waves * get_sm_count();
    constexpr int kBlockSize = 128;
    int64_t total_bytes = static_cast<int64_t>(task_map_len) * sizeof(int4);
    int bytes_per_block = kBlockSize * static_cast<int>(sizeof(int4)) * 8;
    int tail_blocks = static_cast<int>((total_bytes + bytes_per_block - 1) / bytes_per_block);
    if (tail_blocks < 1) {
      tail_blocks = 1;
    }
    if (tail_blocks > 32) {
      tail_blocks = 32;
    }
    int grid = num_group + tail_blocks;
    if constexpr (kUsePDL) {
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(grid);
      cfg.blockDim = dim3(kBlockSize);
      cfg.dynamicSmemBytes = 0;
      cfg.stream = stream;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, kernels::build_gated_task_map_kernel<true>,
                         static_cast<int4 *>(task_map_ptr), static_cast<const int *>(cu_tiles_ptr),
                         static_cast<const int *>(tiles_ptr), num_group, num_tile_n, task_map_len);
    } else {
      kernels::build_gated_task_map_kernel<false><<<grid, kBlockSize, 0, stream>>>(
          static_cast<int4 *>(task_map_ptr), static_cast<const int *>(cu_tiles_ptr),
          static_cast<const int *>(tiles_ptr), num_group, num_tile_n, task_map_len);
    }
  }
  int shm_size = (cosize(typename Config::SLayoutX{}) + 2 * cosize(typename Config::SLayoutW{})) *
                 sizeof(Tin) +
                 cosize(SLayoutD{}) * sizeof(Tin) +
                 (kTaskLoopPolicy == 0 ? sizeof(int4) * num_waves
                                        : sizeof(int) * (num_group + 1));
  auto kernel = kernels::group_gated_gemm_fp8_kernel<
      Config, decltype(tma_a), decltype(tma_b_gate), decltype(tma_b_up), decltype(tma_d),
      kTaskLoopPolicy, kUseBFloat16PrecisionMultiply, kUsePDL>;
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
  cudaLaunchKernelEx(&cfg, kernel, tma_b_gate, tma_b_up, tma_d,
                     static_cast<cute::TmaDescriptor *>(tmas_ptr),
                     static_cast<const int *>(seqlens_ptr),
                     static_cast<const int *>(cu_seqlens_ptr),
                     static_cast<const float *>(gate_up_scale_ptr),
                     static_cast<const float *>(act_scale_ptr), static_cast<int *>(tiles_ptr),
                     static_cast<int *>(cu_tiles_ptr), static_cast<int4 *>(task_map_ptr),
                     static_cast<Tin *>(y_ptr), num_group, m, n, k, flat_divider);
}

void group_gated_gemm_fp8_async(
    void *y_ptr, const void *x_ptr, const void *gate_up_weight_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *gate_up_scale_ptr, const void *act_scale_ptr, void *tmas_ptr,
    void *tiles_ptr, void *cu_tiles_ptr, void *task_map_ptr, int num_waves,
    int num_group, int m, int n, int k,
    int num_seq_per_group_avg, bool use_bf16_mul, bool use_pdl, cudaStream_t stream) {
  use_pdl = true;
#define LAUNCH(TM, ST, POLICY, BF, PDL)                                                \
  launch_group_gated_gemm_fp8<TM, ST, POLICY, BF, PDL>(                               \
      y_ptr, x_ptr, gate_up_weight_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_scale_ptr,\
      act_scale_ptr, tmas_ptr, tiles_ptr, cu_tiles_ptr, task_map_ptr, num_waves,       \
      num_group, m, n, k, stream)
#define DISPATCH_TM(POLICY, BF, PDL)            \
  do {                                          \
    if (num_seq_per_group_avg <= 8)             \
      LAUNCH(8, 5, POLICY, BF, PDL);            \
    else if (num_seq_per_group_avg <= 16)       \
      LAUNCH(16, 5, POLICY, BF, PDL);           \
    else if (num_seq_per_group_avg <= 32)       \
      LAUNCH(32, 4, POLICY, BF, PDL);           \
    else if (num_seq_per_group_avg <= 48)       \
      LAUNCH(48, 4, POLICY, BF, PDL);           \
    else if (num_seq_per_group_avg <= 64)       \
      LAUNCH(64, 3, POLICY, BF, PDL);           \
    else if (num_seq_per_group_avg <= 96)       \
      LAUNCH(48, 4, POLICY, BF, PDL);           \
    else if (num_seq_per_group_avg <= 128)      \
      LAUNCH(32, 4, POLICY, BF, PDL);           \
    else if (num_seq_per_group_avg <= 144)      \
      LAUNCH(48, 4, POLICY, BF, PDL);           \
    else                                        \
      LAUNCH(64, 3, POLICY, BF, PDL);           \
  } while (0)
#define DISPATCH(BF, PDL)                                                            \
  do {                                                                               \
    if (task_map_ptr != nullptr && num_seq_per_group_avg <= 8) {                     \
      DISPATCH_TM(0, BF, PDL);                                                       \
    } else if (k > 1024 && n > 1024 && cu_tiles_ptr != nullptr) {                    \
      DISPATCH_TM(2, BF, PDL);                                                       \
    } else {                                                                         \
      DISPATCH_TM(1, BF, PDL);                                                       \
    }                                                                                \
  } while (0)
  if (use_pdl) {
    if (use_bf16_mul) DISPATCH(true, true); else DISPATCH(false, true);
  } else {
    if (use_bf16_mul) DISPATCH(true, false); else DISPATCH(false, false);
  }
#undef DISPATCH
#undef DISPATCH_TM
#undef LAUNCH
}

}  // namespace group_gemm
}  // namespace hpc
