/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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

#ifndef _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_H_
#define _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_H_

/*
// fence.proxy.async; // 5. PTX ISA 80, SM_90
template <typename = void>
__device__ static inline void fence_proxy_async();
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void fence_proxy_async()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async; // 5." : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// fence.proxy.async.space; // 6. PTX ISA 80, SM_90
// .space     = { .global, .shared::cluster, .shared::cta }
template <cuda::ptx::dot_space Space>
__device__ static inline void fence_proxy_async(
  cuda::ptx::space_t<Space> space);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_space _Space>
_CCCL_DEVICE static inline void fence_proxy_async(::cuda::ptx::space_t<_Space> __space)
{
  static_assert(__space == space_global || __space == space_cluster || __space == space_shared, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__space == space_global)
  {
    asm volatile("fence.proxy.async.global; // 6." : : : "memory");
  }
  else if constexpr (__space == space_cluster)
  {
    asm volatile("fence.proxy.async.shared::cluster; // 6." : : : "memory");
  }
  else if constexpr (__space == space_shared)
  {
    asm volatile("fence.proxy.async.shared::cta; // 6." : : : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_H_
