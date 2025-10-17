/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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

//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_STD___ATOMIC_SCOPES_H
#define __CUDA_STD___ATOMIC_SCOPES_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// REMEMBER CHANGES TO THESE ARE ABI BREAKING
// TODO: Space values out for potential new scopes
#ifndef __ATOMIC_BLOCK
#  define __ATOMIC_SYSTEM 0 // 0 indicates default
#  define __ATOMIC_DEVICE 1
#  define __ATOMIC_BLOCK  2
#  define __ATOMIC_THREAD 10
#endif //__ATOMIC_BLOCK

enum thread_scope
{
  thread_scope_system = __ATOMIC_SYSTEM,
  thread_scope_device = __ATOMIC_DEVICE,
  thread_scope_block  = __ATOMIC_BLOCK,
  thread_scope_thread = __ATOMIC_THREAD
};

struct __thread_scope_thread_tag
{};
struct __thread_scope_block_tag
{};
struct __thread_scope_cluster_tag
{};
struct __thread_scope_device_tag
{};
struct __thread_scope_system_tag
{};

template <int _Scope>
struct __scope_enum_to_tag
{};
/* This would be the implementation once an actual thread-scope backend exists.
template<> struct __scope_enum_to_tag<(int)thread_scope_thread> {
    using type = __thread_scope_thread_tag; };
Until then: */
template <>
struct __scope_enum_to_tag<(int) thread_scope_thread>
{
  using __tag = __thread_scope_block_tag;
};
template <>
struct __scope_enum_to_tag<(int) thread_scope_block>
{
  using __tag = __thread_scope_block_tag;
};
template <>
struct __scope_enum_to_tag<(int) thread_scope_device>
{
  using __tag = __thread_scope_device_tag;
};
template <>
struct __scope_enum_to_tag<(int) thread_scope_system>
{
  using __tag = __thread_scope_system_tag;
};

template <int _Scope>
using __scope_to_tag = typename __scope_enum_to_tag<_Scope>::__tag;

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_CUDA

using ::cuda::std::thread_scope;
using ::cuda::std::thread_scope_block;
using ::cuda::std::thread_scope_device;
using ::cuda::std::thread_scope_system;
using ::cuda::std::thread_scope_thread;

using ::cuda::std::__thread_scope_block_tag;
using ::cuda::std::__thread_scope_device_tag;
using ::cuda::std::__thread_scope_system_tag;

_CCCL_END_NAMESPACE_CUDA

#include <uscl/std/__cccl/epilogue.h>

#endif // __CUDA_STD___ATOMIC_SCOPES_H
