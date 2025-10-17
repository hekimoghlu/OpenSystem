/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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

#ifndef __CUDA_STD___ATOMIC_PLATFORM_H
#define __CUDA_STD___ATOMIC_PLATFORM_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_COMPILER(MSVC)
#  include <cuda/std/__atomic/platform/msvc_to_builtins.h>
#endif

#if defined(__CLANG_ATOMIC_BOOL_LOCK_FREE)
#  define LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE     __CLANG_ATOMIC_BOOL_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE     __CLANG_ATOMIC_CHAR_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE __CLANG_ATOMIC_CHAR16_T_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE __CLANG_ATOMIC_CHAR32_T_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE  __CLANG_ATOMIC_WCHAR_T_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE    __CLANG_ATOMIC_SHORT_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_INT_LOCK_FREE      __CLANG_ATOMIC_INT_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_LONG_LOCK_FREE     __CLANG_ATOMIC_LONG_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE    __CLANG_ATOMIC_LLONG_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE  __CLANG_ATOMIC_POINTER_LOCK_FREE
#elif defined(__GCC_ATOMIC_BOOL_LOCK_FREE)
#  define LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE     __GCC_ATOMIC_BOOL_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE     __GCC_ATOMIC_CHAR_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE __GCC_ATOMIC_CHAR16_T_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE __GCC_ATOMIC_CHAR32_T_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE  __GCC_ATOMIC_WCHAR_T_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE    __GCC_ATOMIC_SHORT_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_INT_LOCK_FREE      __GCC_ATOMIC_INT_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_LONG_LOCK_FREE     __GCC_ATOMIC_LONG_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE    __GCC_ATOMIC_LLONG_LOCK_FREE
#  define LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE  __GCC_ATOMIC_POINTER_LOCK_FREE
#else // !defined(__CLANG_ATOMIC_BOOL_LOCK_FREE) && !defined(__GCC_ATOMIC_BOOL_LOCK_FREE)
#  define LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE     2
#  define LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE     2
#  define LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE 2
#  define LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE 2
#  define LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE  2
#  define LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE    2
#  define LIBCUDACXX_ATOMIC_INT_LOCK_FREE      2
#  define LIBCUDACXX_ATOMIC_LONG_LOCK_FREE     2
#  define LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE    2
#  define LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE  2
#endif

#define _LIBCUDACXX_ATOMIC_IS_LOCK_FREE(size) (size <= 8)

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)
template <typename _Tp>
struct __atomic_is_always_lock_free
{
  enum
  {
    __value = _LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0)
  };
};
#else
template <typename _Tp>
struct __atomic_is_always_lock_free
{
  enum
  {
    __value = sizeof(_Tp) <= 8
  };
};
#endif // defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // __CUDA_STD___ATOMIC_PLATFORM_H
