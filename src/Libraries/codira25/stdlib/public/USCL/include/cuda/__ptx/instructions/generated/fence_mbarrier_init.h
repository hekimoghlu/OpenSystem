/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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

#ifndef _CUDA_PTX_GENERATED_FENCE_MBARRIER_INIT_H_
#define _CUDA_PTX_GENERATED_FENCE_MBARRIER_INIT_H_

/*
// fence.mbarrier_init.sem.scope; // 3. PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
template <typename = void>
__device__ static inline void fence_mbarrier_init(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_mbarrier_init_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void fence_mbarrier_init(::cuda::ptx::sem_release_t, ::cuda::ptx::scope_cluster_t)
{
// __sem == sem_release (due to parameter type constraint)
// __scope == scope_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("fence.mbarrier_init.release.cluster; // 3." : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_mbarrier_init_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_FENCE_MBARRIER_INIT_H_
