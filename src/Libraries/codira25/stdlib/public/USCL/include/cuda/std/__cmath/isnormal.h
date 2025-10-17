/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#ifndef _CUDA_STD___CMATH_ISNORMAL_H
#define _CUDA_STD___CMATH_ISNORMAL_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__cmath/fpclassify.h>
#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__floating_point/cuda_fp_types.h>
#include <uscl/std/__type_traits/is_integral.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_CHECK_BUILTIN(builtin_isnormal) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISNORMAL(...) __builtin_isnormal(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(isnormal)

// nvcc does not implement __builtin_isnormal
#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_ISNORMAL
#endif // _CCCL_CUDA_COMPILER(NVCC)

[[nodiscard]] _CCCL_API constexpr bool isnormal(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISNORMAL)
  return _CCCL_BUILTIN_ISNORMAL(__x);
#else // ^^^ _CCCL_BUILTIN_ISNORMAL ^^^ / vvv !_CCCL_BUILTIN_ISNORMAL vvv
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
#endif // !_CCCL_BUILTIN_ISNORMAL
}

[[nodiscard]] _CCCL_API constexpr bool isnormal(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISNORMAL)
  return _CCCL_BUILTIN_ISNORMAL(__x);
#else // ^^^ _CCCL_BUILTIN_ISNORMAL ^^^ / vvv !_CCCL_BUILTIN_ISNORMAL vvv
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
#endif // !_CCCL_BUILTIN_ISNORMAL
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API constexpr bool isnormal(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ISNORMAL)
  return _CCCL_BUILTIN_ISNORMAL(__x);
#  else // ^^^ _CCCL_BUILTIN_ISNORMAL ^^^ / vvv !_CCCL_BUILTIN_ISNORMAL vvv
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
#  endif // !_CCCL_BUILTIN_ISNORMAL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _CCCL_HAS_NVFP16()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__half __x) noexcept
{
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_bfloat16 __x) noexcept
{
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp8_e4m3 __x) noexcept
{
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp8_e5m2 __x) noexcept
{
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp8_e8m0 __x) noexcept
{
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp6_e2m3 __x) noexcept
{
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp6_e3m2 __x) noexcept
{
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp4_e2m1 __x) noexcept
{
  return ::cuda::std::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_integral_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool isnormal(_Tp __x) noexcept
{
  return __x != 0;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_ISNORMAL_H
