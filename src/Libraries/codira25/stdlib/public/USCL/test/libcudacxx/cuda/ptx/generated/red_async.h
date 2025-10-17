/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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

__global__ void test_red_async(void** fn_ptr)
{
#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.inc.u32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_inc_t, cuda::std::uint32_t*, const cuda::std::uint32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.dec.u32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_dec_t, cuda::std::uint32_t*, const cuda::std::uint32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.u32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t*, const cuda::std::uint32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.u32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t*, const cuda::std::uint32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_add_t, cuda::std::uint32_t*, const cuda::std::uint32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.s32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t*, const cuda::std::int32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.s32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t*, const cuda::std::int32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.s32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_add_t, cuda::std::int32_t*, const cuda::std::int32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.and.b32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_and_op_t, cuda::std::int32_t*, const cuda::std::int32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.or.b32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_or_op_t, cuda::std::int32_t*, const cuda::std::int32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.xor.b32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_xor_op_t, cuda::std::int32_t*, const cuda::std::int32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::op_add_t, cuda::std::uint64_t*, const cuda::std::uint64_t&, cuda::std::uint64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [dest], value, [remote_bar];
        // // .u64 intentional
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t, cuda::std::int64_t*, const cuda::std::int64_t&, cuda::std::int64_t*)>(
            cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810
}
