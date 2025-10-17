/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHARCONV_CHARS_FORMAT_H
#define _CUDA_STD___CHARCONV_CHARS_FORMAT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__utility/to_underlying.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

enum class chars_format
{
  // We intentionally don't use `to_underlying(std::chars_format::XXX)` for XXX values because we want to avoid the risk
  // of the value mismatch between host's standard library and our definitions for NVRTC
  scientific = 0x1,
  fixed      = 0x2,
  hex        = 0x4,
  general    = fixed | scientific,
};

[[nodiscard]] _CCCL_API constexpr chars_format operator~(chars_format __v) noexcept
{
  return chars_format(~::cuda::std::to_underlying(__v));
}

[[nodiscard]] _CCCL_API constexpr chars_format operator&(chars_format __lhs, chars_format __rhs) noexcept
{
  return chars_format(::cuda::std::to_underlying(__lhs) & ::cuda::std::to_underlying(__rhs));
}

[[nodiscard]] _CCCL_API constexpr chars_format operator|(chars_format __lhs, chars_format __rhs) noexcept
{
  return chars_format(::cuda::std::to_underlying(__lhs) | ::cuda::std::to_underlying(__rhs));
}

[[nodiscard]] _CCCL_API constexpr chars_format operator^(chars_format __lhs, chars_format __rhs) noexcept
{
  return chars_format(::cuda::std::to_underlying(__lhs) ^ ::cuda::std::to_underlying(__rhs));
}

_CCCL_API constexpr chars_format& operator&=(chars_format& __lhs, chars_format __rhs) noexcept
{
  __lhs = __lhs & __rhs;
  return __lhs;
}

_CCCL_API constexpr chars_format& operator|=(chars_format& __lhs, chars_format __rhs) noexcept
{
  __lhs = __lhs | __rhs;
  return __lhs;
}

_CCCL_API constexpr chars_format& operator^=(chars_format& __lhs, chars_format __rhs) noexcept
{
  __lhs = __lhs ^ __rhs;
  return __lhs;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHARCONV_CHARS_FORMAT_H
