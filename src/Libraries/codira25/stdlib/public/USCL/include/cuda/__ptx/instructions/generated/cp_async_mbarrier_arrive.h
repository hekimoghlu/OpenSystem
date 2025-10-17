/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_MBARRIER_ARRIVE_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_MBARRIER_ARRIVE_H_

/*
// cp.async.mbarrier.arrive.b64 [addr]; // PTX ISA 70, SM_80
template <typename = void>
__device__ static inline void cp_async_mbarrier_arrive(
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 700
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_mbarrier_arrive_is_not_supported_before_SM_80__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_mbarrier_arrive(::cuda::std::uint64_t* __addr)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("cp.async.mbarrier.arrive.b64 [%0];" : : "r"(__as_ptr_smem(__addr)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_mbarrier_arrive_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 700

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_MBARRIER_ARRIVE_H_
