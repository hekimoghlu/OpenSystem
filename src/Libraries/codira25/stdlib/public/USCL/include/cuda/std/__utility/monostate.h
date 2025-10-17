/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_MONOSTATE_H
#define _CUDA_STD___UTILITY_MONOSTATE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/ordering.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <uscl/std/__functional/hash.h>
#include <uscl/std/cstddef>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT monostate
{};

_CCCL_API constexpr bool operator==(monostate, monostate) noexcept
{
  return true;
}

#if _CCCL_STD_VER < 2020

_CCCL_API constexpr bool operator!=(monostate, monostate) noexcept
{
  return false;
}

#endif // _CCCL_STD_VER < 2020

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

_CCCL_API constexpr strong_ordering operator<=>(monostate, monostate) noexcept
{
  return strong_ordering::equal;
}

#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

_CCCL_API constexpr bool operator<(monostate, monostate) noexcept
{
  return false;
}

_CCCL_API constexpr bool operator>(monostate, monostate) noexcept
{
  return false;
}

_CCCL_API constexpr bool operator<=(monostate, monostate) noexcept
{
  return true;
}

_CCCL_API constexpr bool operator>=(monostate, monostate) noexcept
{
  return true;
}

#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#ifndef __cuda_std__
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT hash<monostate>
{
  using argument_type = monostate;
  using result_type   = size_t;

  _CCCL_API inline result_type operator()(const argument_type&) const noexcept
  {
    return 66740831; // return a fundamentally attractive random value.
  }
};
#endif // __cuda_std__

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_MONOSTATE_H
