/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_MULTICAST_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_MULTICAST_H_

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem], size, [smem_bar], ctaMask;
// PTX ISA 80, SM_90a, SM_100a, SM_110a
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_is_only_supported_on_SM_90a_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size,
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100)))
  asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__as_ptr_gmem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_is_only_supported_on_SM_90a_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_MULTICAST_H_
