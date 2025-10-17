/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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

#ifndef _CUDA_STD___CHARCONV_TO_CHARS_H
#define _CUDA_STD___CHARCONV_TO_CHARS_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/__cmath/uabs.h>
#include <uscl/std/__charconv/to_chars_result.h>
#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__cstddef/types.h>
#include <uscl/std/__type_traits/conditional.h>
#include <uscl/std/__type_traits/is_integer.h>
#include <uscl/std/__type_traits/is_same.h>
#include <uscl/std/__type_traits/is_signed.h>
#include <uscl/std/__type_traits/make_unsigned.h>
#include <uscl/std/cstdint>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[nodiscard]] _CCCL_API constexpr char __to_chars_value_to_char(int __v, int __base) noexcept
{
  _CCCL_ASSERT(__v >= 0 && __v < __base, "value must be in the range [0, base)");
  const int __offset = (__base < 10 || __v < 10) ? '0' : ('a' - 10);
  return static_cast<char>(__offset + __v);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr int __to_chars_int_width(_Tp __v, int __base) noexcept
{
  using _Up = ::cuda::std::conditional_t<sizeof(_Tp) >= sizeof(uint32_t), make_unsigned_t<_Tp>, uint32_t>;

  auto __uv = static_cast<_Up>(__v);

  const auto __ubase   = static_cast<_Up>(__base);
  const auto __ubase_2 = __ubase * __ubase;
  const auto __ubase_3 = __ubase_2 * __ubase;
  const auto __ubase_4 = __ubase_2 * __ubase_2;

  int __r = 0;
  while (true)
  {
    if (__uv < __ubase)
    {
      return __r + 1;
    }
    else if (__uv < __ubase_2)
    {
      return __r + 2;
    }
    else if (__uv < __ubase_3)
    {
      return __r + 3;
    }
    else if (__uv < __ubase_4)
    {
      return __r + 4;
    }

    __uv /= __ubase_4;
    __r += 4;
  }

  _CCCL_UNREACHABLE();
}

template <class _Tp>
_CCCL_API constexpr void __to_chars_int_generic(char* __last, _Tp __value, int __base) noexcept
{
  do
  {
    const int __c = __value % __base;
    *--__last     = ::cuda::std::__to_chars_value_to_char(__c, __base);
    __value /= static_cast<_Tp>(__base);
  } while (__value != 0);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr to_chars_result
to_chars(char* __first, char* __last, _Tp __value, int __base = 10) noexcept
{
  _CCCL_ASSERT(__base >= 2 && __base <= 36, "base must be in the range [2, 36]");
  _CCCL_ASSERT(__first <= __last, "output range must be a valid range");

  if constexpr (is_signed_v<_Tp>)
  {
    if (__value < _Tp{0} && __first < __last)
    {
      *__first++ = '-';
    }
    return ::cuda::std::to_chars(__first, __last, ::cuda::uabs(__value), __base);
  }
  else
  {
    const ptrdiff_t __cap = __last - __first;
    const int __n         = ::cuda::std::__to_chars_int_width(__value, __base);

    if (__n > __cap)
    {
      return {__last, errc::value_too_large};
    }

    char* __new_last = __first + __n;

    ::cuda::std::__to_chars_int_generic(__new_last, __value, __base);

    return {__new_last, errc{}};
  }
}

[[nodiscard]] _CCCL_API constexpr to_chars_result
to_chars(char* __first, char* __last, char __value, int __base = 10) noexcept
{
  if constexpr (is_signed_v<char>)
  {
    return ::cuda::std::to_chars(__first, __last, static_cast<signed char>(__value), __base);
  }
  else
  {
    return ::cuda::std::to_chars(__first, __last, static_cast<unsigned char>(__value), __base);
  }
}

_CCCL_API constexpr to_chars_result to_chars(char*, char*, bool, int = 10) noexcept = delete;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHARCONV_TO_CHARS_H
