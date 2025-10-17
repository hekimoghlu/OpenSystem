/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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

#ifndef _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_GENERIC_SYNC_RESTRICT_H_
#define _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_GENERIC_SYNC_RESTRICT_H_

/*
// fence.proxy.async::generic.sem.sync_restrict::space.scope; // PTX ISA 86, SM_90
// .sem       = { .acquire }
// .space     = { .shared::cluster }
// .scope     = { .cluster }
template <typename = void>
__device__ static inline void fence_proxy_async_generic_sync_restrict(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::space_cluster_t,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_async_generic_sync_restrict_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void fence_proxy_async_generic_sync_restrict(
  ::cuda::ptx::sem_acquire_t, ::cuda::ptx::space_cluster_t, ::cuda::ptx::scope_cluster_t)
{
// __sem == sem_acquire (due to parameter type constraint)
// __space == space_cluster (due to parameter type constraint)
// __scope == scope_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_proxy_async_generic_sync_restrict_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// fence.proxy.async::generic.sem.sync_restrict::space.scope; // PTX ISA 86, SM_90
// .sem       = { .release }
// .space     = { .shared::cta }
// .scope     = { .cluster }
template <typename = void>
__device__ static inline void fence_proxy_async_generic_sync_restrict(
  cuda::ptx::sem_release_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_async_generic_sync_restrict_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void fence_proxy_async_generic_sync_restrict(
  ::cuda::ptx::sem_release_t, ::cuda::ptx::space_shared_t, ::cuda::ptx::scope_cluster_t)
{
// __sem == sem_release (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __scope == scope_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_proxy_async_generic_sync_restrict_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_GENERIC_SYNC_RESTRICT_H_
