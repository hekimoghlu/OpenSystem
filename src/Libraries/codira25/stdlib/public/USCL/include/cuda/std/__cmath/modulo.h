/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CMATH_MODULO_H
#define _CUDA_STD___CMATH_MODULO_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__cmath/copysign.h>
#include <uscl/std/__cmath/rounding_functions.h>
#include <uscl/std/__floating_point/cuda_fp_types.h>
#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/is_arithmetic.h>
#include <uscl/std/__type_traits/is_same.h>
#include <uscl/std/__type_traits/promote.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// fmod

#if _CCCL_CHECK_BUILTIN(builtin_fmod) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FMODF(...) __builtin_fmodf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMOD(...)  __builtin_fmod(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMODL(...) __builtin_fmodl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fmod)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "modf"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_FMODF
#  undef _CCCL_BUILTIN_FMOD
#  undef _CCCL_BUILTIN_FMODL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float fmod(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_FMODF)
  return _CCCL_BUILTIN_FMODF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FMODF ^^^ / vvv !_CCCL_BUILTIN_FMODF vvv
  return ::fmodf(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_FMODF ^^^
}

[[nodiscard]] _CCCL_API inline float fmodf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_FMODF)
  return _CCCL_BUILTIN_FMODF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FMODF ^^^ / vvv !_CCCL_BUILTIN_FMODF vvv
  return ::fmodf(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_FMODF ^^^
}

[[nodiscard]] _CCCL_API inline double fmod(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_FMOD)
  return _CCCL_BUILTIN_FMOD(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FMOD ^^^ / vvv !_CCCL_BUILTIN_FMOD vvv
  return ::fmod(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_FMOD ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double fmod(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_FMODL)
  return _CCCL_BUILTIN_FMODL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_FMODL ^^^ / vvv !_CCCL_BUILTIN_FMODL vvv
  return ::fmodl(__x, __y);
#  endif // ^^^ !_CCCL_BUILTIN_FMODL ^^^
}

[[nodiscard]] _CCCL_API inline long double fmodl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_FMODL)
  return _CCCL_BUILTIN_FMODL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_FMODL ^^^ / vvv !_CCCL_BUILTIN_FMODL vvv
  return ::fmodl(__x, __y);
#  endif // ^^^ !_CCCL_BUILTIN_FMODL ^^^
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half fmod(__half __x, __half __y) noexcept
{
  return ::__float2half(::cuda::std::fmod(::__half2float(__x), ::__half2float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 fmod(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return ::__float2bfloat16(::cuda::std::fmod(::__bfloat162float(__x), ::__bfloat162float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<is_arithmetic_v<_A1> && is_arithmetic_v<_A2>, int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, _A2> fmod(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(is_same_v<_A1, __result_type> && is_same_v<_A2, __result_type>) );
  return ::cuda::std::fmod((__result_type) __x, (__result_type) __y);
}

// modf

#if _CCCL_CHECK_BUILTIN(builtin_modf) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_MODFF(...) __builtin_modff(__VA_ARGS__)
#  define _CCCL_BUILTIN_MODF(...)  __builtin_modf(__VA_ARGS__)
#  define _CCCL_BUILTIN_MODFL(...) __builtin_modfl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_modf)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "modf"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_MODFF
#  undef _CCCL_BUILTIN_MODF
#  undef _CCCL_BUILTIN_MODFL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float modf(float __x, float* __y) noexcept
{
#if defined(_CCCL_BUILTIN_MODFF)
  return _CCCL_BUILTIN_MODFF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_MODFF ^^^ / vvv !_CCCL_BUILTIN_MODFF vvv
  return ::modff(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_MODFF ^^^
}

[[nodiscard]] _CCCL_API inline float modff(float __x, float* __y) noexcept
{
#if defined(_CCCL_BUILTIN_MODFF)
  return _CCCL_BUILTIN_MODFF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_MODFF ^^^ / vvv !_CCCL_BUILTIN_MODFF vvv
  return ::modff(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_MODFF ^^^
}

[[nodiscard]] _CCCL_API inline double modf(double __x, double* __y) noexcept
{
#if defined(_CCCL_BUILTIN_MODF)
  return _CCCL_BUILTIN_MODF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_MODF ^^^ / vvv !_CCCL_BUILTIN_MODF vvv
  return ::modf(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_MODF ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double modf(long double __x, long double* __y) noexcept
{
#  if defined(_CCCL_BUILTIN_MODFL)
  return _CCCL_BUILTIN_MODFL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_MODFL ^^^ / vvv !_CCCL_BUILTIN_MODFL vvv
  return ::modfl(__x, __y);
#  endif // ^^^ !_CCCL_BUILTIN_MODFL ^^^
}

[[nodiscard]] _CCCL_API inline long double modfl(long double __x, long double* __y) noexcept
{
#  if defined(_CCCL_BUILTIN_MODFL)
  return _CCCL_BUILTIN_MODFL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_MODFL ^^^ / vvv !_CCCL_BUILTIN_MODFL vvv
  return ::modfl(__x, __y);
#  endif // ^^^ !_CCCL_BUILTIN_MODFL ^^^
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half modf(__half __x, __half* __y) noexcept
{
  const __half __integral_part = ::cuda::std::trunc(__x);
  *__y                         = __integral_part;
  return ::__heq(__integral_part, __x)
         ? ::cuda::std::copysign(::__float2half(0.0f), __x)
         : ::__hsub(__x, __integral_part);
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 modf(__nv_bfloat16 __x, __nv_bfloat16* __y) noexcept
{
  const __nv_bfloat16 __integral_part = ::cuda::std::trunc(__x);
  *__y                                = __integral_part;
  return ::__heq(__integral_part, __x)
         ? ::cuda::std::copysign(::__float2bfloat16(0.0f), __x)
         : ::__hsub(__x, __integral_part);
}
#endif // _LIBCUDACXX_HAS_NVBF16()

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_MODULO_H
