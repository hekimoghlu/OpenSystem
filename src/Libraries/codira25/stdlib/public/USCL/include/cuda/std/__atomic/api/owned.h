/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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

#ifndef __CUDA_STD___ATOMIC_API_OWNED_H
#define __CUDA_STD___ATOMIC_API_OWNED_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__atomic/api/common.h>
#include <uscl/std/__atomic/order.h>
#include <uscl/std/__atomic/scopes.h>
#include <uscl/std/__atomic/types.h>
#include <uscl/std/__atomic/wait/notify_wait.h>
#include <uscl/std/__atomic/wait/polling.h>
#include <uscl/std/__type_traits/conditional.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <typename _Tp, typename _Sco>
struct __atomic_common
{
  _CCCL_API constexpr __atomic_common(_Tp __v)
      : __a(__v)
  {}

  _CCCL_HIDE_FROM_ABI constexpr __atomic_common() = default;

  __atomic_storage_t<_Tp> __a;

#if defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)
  static constexpr bool is_always_lock_free = _LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0);
#endif // defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)

  _LIBCUDACXX_ATOMIC_COMMON_IMPL(, )
  _LIBCUDACXX_ATOMIC_COMMON_IMPL(, volatile)
};

template <typename _Tp, typename _Sco>
struct __atomic_arithmetic
{
  _CCCL_API constexpr __atomic_arithmetic(_Tp __v)
      : __a(__v)
  {}

  _CCCL_HIDE_FROM_ABI constexpr __atomic_arithmetic() = default;

  __atomic_storage_t<_Tp> __a;

#if defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)
  static constexpr bool is_always_lock_free = _LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0);
#endif // defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)

  _LIBCUDACXX_ATOMIC_COMMON_IMPL(, )
  _LIBCUDACXX_ATOMIC_COMMON_IMPL(, volatile)

  _LIBCUDACXX_ATOMIC_ARITHMETIC_IMPL(, )
  _LIBCUDACXX_ATOMIC_ARITHMETIC_IMPL(, volatile)
};

template <typename _Tp, typename _Sco>
struct __atomic_bitwise
{
  _CCCL_API constexpr __atomic_bitwise(_Tp __v)
      : __a(__v)
  {}

  _CCCL_HIDE_FROM_ABI constexpr __atomic_bitwise() = default;

  __atomic_storage_t<_Tp> __a;

#if defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)
  static constexpr bool is_always_lock_free = _LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0);
#endif // defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)

  _LIBCUDACXX_ATOMIC_COMMON_IMPL(, )
  _LIBCUDACXX_ATOMIC_COMMON_IMPL(, volatile)

  _LIBCUDACXX_ATOMIC_ARITHMETIC_IMPL(, )
  _LIBCUDACXX_ATOMIC_ARITHMETIC_IMPL(, volatile)

  _LIBCUDACXX_ATOMIC_BITWISE_IMPL(, )
  _LIBCUDACXX_ATOMIC_BITWISE_IMPL(, volatile)
};

template <typename _Tp, typename _Sco>
struct __atomic_pointer
{
  _CCCL_API constexpr __atomic_pointer(_Tp __v)
      : __a(__v)
  {}

  _CCCL_HIDE_FROM_ABI constexpr __atomic_pointer() = default;

  __atomic_storage_t<_Tp> __a;

#if defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)
  static constexpr bool is_always_lock_free = _LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0);
#endif // defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)

  _LIBCUDACXX_ATOMIC_COMMON_IMPL(, )
  _LIBCUDACXX_ATOMIC_COMMON_IMPL(, volatile)

  _LIBCUDACXX_ATOMIC_POINTER_IMPL(, )
  _LIBCUDACXX_ATOMIC_POINTER_IMPL(, volatile)
};

template <typename _Tp, thread_scope _Sco = thread_scope_system>
using __atomic_impl =
  _If<is_pointer<_Tp>::value,
      __atomic_pointer<_Tp, __scope_to_tag<_Sco>>,
      _If<is_floating_point<_Tp>::value,
          __atomic_arithmetic<_Tp, __scope_to_tag<_Sco>>,
          _If<is_integral<_Tp>::value,
              __atomic_bitwise<_Tp, __scope_to_tag<_Sco>>,
              __atomic_common<_Tp, __scope_to_tag<_Sco>>>>>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // __CUDA_STD___ATOMIC_API_OWNED_H
