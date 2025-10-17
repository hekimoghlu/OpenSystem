/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CSTDLIB_DIV_H
#define _CUDA_STD___CSTDLIB_DIV_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdlib>
#endif // !_CCCL_COMPILER(NVRTC)

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// If available, use the host's div_t, ldiv_t, and lldiv_t types because the struct members order is
// implementation-defined.
#if !_CCCL_COMPILER(NVRTC)
using ::div_t;
using ::ldiv_t;
using ::lldiv_t;
#else // ^^^ !_CCCL_COMPILER(NVRTC) / _CCCL_COMPILER(NVRTC) vvv
struct _CCCL_TYPE_VISIBILITY_DEFAULT div_t
{
  int quot;
  int rem;
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT ldiv_t
{
  long quot;
  long rem;
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT lldiv_t
{
  long long quot;
  long long rem;
};
#endif // !_CCCL_COMPILER(NVRTC)

[[nodiscard]] _CCCL_API constexpr div_t div(int __x, int __y) noexcept
{
  div_t __result{};
  __result.quot = __x / __y;
  __result.rem  = __x % __y;
  return __result;
}

[[nodiscard]] _CCCL_API constexpr ldiv_t ldiv(long __x, long __y) noexcept
{
  ldiv_t __result{};
  __result.quot = __x / __y;
  __result.rem  = __x % __y;
  return __result;
}

[[nodiscard]] _CCCL_API constexpr ldiv_t div(long __x, long __y) noexcept
{
  return ::cuda::std::ldiv(__x, __y);
}

[[nodiscard]] _CCCL_API constexpr lldiv_t lldiv(long long __x, long long __y) noexcept
{
  lldiv_t __result{};
  __result.quot = __x / __y;
  __result.rem  = __x % __y;
  return __result;
}

[[nodiscard]] _CCCL_API constexpr lldiv_t div(long long __x, long long __y) noexcept
{
  return ::cuda::std::lldiv(__x, __y);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CSTDLIB_DIV_H
