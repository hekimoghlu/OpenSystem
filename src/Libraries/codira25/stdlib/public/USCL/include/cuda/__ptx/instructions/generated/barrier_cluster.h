/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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

#ifndef _CUDA_PTX_GENERATED_BARRIER_CLUSTER_H_
#define _CUDA_PTX_GENERATED_BARRIER_CLUSTER_H_

/*
// barrier.cluster.arrive; // PTX ISA 78, SM_90
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_arrive();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_arrive()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// barrier.cluster.wait; // PTX ISA 78, SM_90
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_wait();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_wait()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.wait;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
// .sem       = { .release }
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_release_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(::cuda::ptx::sem_release_t)
{
// __sem == sem_release (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
// .sem       = { .relaxed }
// Marked volatile
template <typename = void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_relaxed_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(::cuda::ptx::sem_relaxed_t)
{
// __sem == sem_relaxed (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.relaxed;" : : :);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// barrier.cluster.wait.sem; // PTX ISA 80, SM_90
// .sem       = { .acquire }
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_wait(
  cuda::ptx::sem_acquire_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_wait(::cuda::ptx::sem_acquire_t)
{
// __sem == sem_acquire (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.wait.acquire;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_BARRIER_CLUSTER_H_
