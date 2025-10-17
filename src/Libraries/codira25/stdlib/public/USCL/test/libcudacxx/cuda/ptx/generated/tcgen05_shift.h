/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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

__global__ void test_tcgen05_shift(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.shift.cta_group::1.down [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t)>(cuda::ptx::tcgen05_shift_down));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.shift.cta_group::1.down [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t)>(cuda::ptx::tcgen05_shift_down));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.shift.cta_group::1.down [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t)>(cuda::ptx::tcgen05_shift_down));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.shift.cta_group::2.down [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t)>(cuda::ptx::tcgen05_shift_down));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.shift.cta_group::2.down [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t)>(cuda::ptx::tcgen05_shift_down));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.shift.cta_group::2.down [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t)>(cuda::ptx::tcgen05_shift_down));));
#endif // __cccl_ptx_isa >= 860
}
