/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
#ifndef _CUDA_STD___CMATH_COPYSIGN_H
#define _CUDA_STD___CMATH_COPYSIGN_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__floating_point/fp.h>
#include <uscl/std/__type_traits/is_extended_arithmetic.h>
#include <uscl/std/__type_traits/is_integral.h>
#include <uscl/std/limits>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr auto copysign(_Tp __x, [[maybe_unused]] _Tp __y) noexcept
{
  if constexpr (is_integral_v<_Tp>)
  {
    if constexpr (!numeric_limits<_Tp>::is_signed)
    {
      return static_cast<double>(__x);
    }
    else
    {
      const auto __x_dbl = static_cast<double>(__x);
      if (__y < 0)
      {
        return (__x < 0) ? __x_dbl : -__x_dbl;
      }
      else
      {
        return (__x < 0) ? -__x_dbl : __x_dbl;
      }
    }
  }
  else // ^^^ integral ^^^ / vvv floating_point vvv
  {
    if constexpr (!numeric_limits<_Tp>::is_signed)
    {
      return __x;
    }
    else
    {
      const auto __val = (::cuda::std::__fp_get_storage(__x) & __fp_exp_mant_mask_of_v<_Tp>)
                       | (::cuda::std::__fp_get_storage(__y) & __fp_sign_mask_of_v<_Tp>);
      return ::cuda::std::__fp_from_storage<_Tp>(static_cast<__fp_storage_of_t<_Tp>>(__val));
    }
  }
}

[[nodiscard]] _CCCL_API constexpr float copysignf(float __x, float __y) noexcept
{
  return ::cuda::std::copysign(__x, __y);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API constexpr long double copysignl(long double __x, long double __y) noexcept
{
  return ::cuda::std::copysign(__x, __y);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_COPYSIGN_H
