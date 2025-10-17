/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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

// We use a special strategy to force the generation of the PTX. This is mainly
// a fight against dead-code-elimination in the NVVM layer.
//
// The reason we need this strategy is because certain older versions of ptxas
// segfault when a non-sensical sequence of PTX is generated. So instead, we try
// to force the instantiation and compilation to PTX of all the overloads of the
// PTX wrapping functions.
//
// We do this by writing a function pointer of each overload to the kernel
// parameter `fn_ptr`.
//
// Because `fn_ptr` is possibly visible outside this translation unit, the
// compiler must compile all the functions which are stored.

__global__ void test_barrier_cluster(void** fn_ptr)
{
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // barrier.cluster.arrive;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::barrier_cluster_arrive));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // barrier.cluster.wait;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::barrier_cluster_wait));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // barrier.cluster.arrive.release;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::sem_release_t)>(cuda::ptx::barrier_cluster_arrive));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // barrier.cluster.arrive.relaxed;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::sem_relaxed_t)>(cuda::ptx::barrier_cluster_arrive));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // barrier.cluster.wait.acquire;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::sem_acquire_t)>(cuda::ptx::barrier_cluster_wait));));
#endif // __cccl_ptx_isa >= 800
}
