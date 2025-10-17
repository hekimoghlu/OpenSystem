/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

__global__ void test_fence(void** fn_ptr)
{
#if __cccl_ptx_isa >= 600
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // fence.sc.cta; // 1.
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
          // fence.sc.gpu; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
          // fence.sc.sys; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));));
#endif // __cccl_ptx_isa >= 600

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // fence.sc.cluster; // 2.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_cluster_t)>(cuda::ptx::fence));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 600
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // fence.acq_rel.cta; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
          // fence.acq_rel.gpu; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
          // fence.acq_rel.sys; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));));
#endif // __cccl_ptx_isa >= 600

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // fence.acq_rel.cluster; // 2.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_cluster_t)>(cuda::ptx::fence));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.acquire.cta;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
          // fence.acquire.cluster;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t)>(cuda::ptx::fence));
          // fence.acquire.gpu;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
          // fence.acquire.sys;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.release.cta;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
          // fence.release.cluster;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t)>(cuda::ptx::fence));
          // fence.release.gpu;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
          // fence.release.sys;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));));
#endif // __cccl_ptx_isa >= 860
}
