/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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

#ifndef _CUDA_PTX_GENERATED_MBARRIER_INIT_H_
#define _CUDA_PTX_GENERATED_MBARRIER_INIT_H_

/*
// mbarrier.init.shared.b64 [addr], count; // PTX ISA 70, SM_80
template <typename = void>
__device__ static inline void mbarrier_init(
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 700
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_init_is_not_supported_before_SM_80__();
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_init(::cuda::std::uint64_t* __addr, const ::cuda::std::uint32_t& __count)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("mbarrier.init.shared.b64 [%0], %1;" : : "r"(__as_ptr_smem(__addr)), "r"(__count) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_init_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 700

#endif // _CUDA_PTX_GENERATED_MBARRIER_INIT_H_
