/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#ifndef _CUDA___CMATH_CEIL_DIV_H
#define _CUDA___CMATH_CEIL_DIV_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__algorithm/min.h>
#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__type_traits/common_type.h>
#include <uscl/std/__type_traits/is_constant_evaluated.h>
#include <uscl/std/__type_traits/is_enum.h>
#include <uscl/std/__type_traits/is_integral.h>
#include <uscl/std/__type_traits/is_signed.h>
#include <uscl/std/__type_traits/make_unsigned.h>
#include <uscl/std/__type_traits/underlying_type.h>
#include <uscl/std/__utility/to_underlying.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND ::cuda::std::is_integral_v<_Up>)
[[nodiscard]] _CCCL_API constexpr ::cuda::std::common_type_t<_Tp, _Up> ceil_div(const _Tp __a, const _Up __b) noexcept
{
  _CCCL_ASSERT(__b > _Up{0}, "cuda::ceil_div: 'b' must be positive");
  if constexpr (::cuda::std::is_signed_v<_Tp>)
  {
    _CCCL_ASSERT(__a >= _Tp{0}, "cuda::ceil_div: 'a' must be non negative");
  }
  using _Common = ::cuda::std::common_type_t<_Tp, _Up>;
  using _Prom   = decltype(_Tp{} / _Up{});
  using _UProm  = ::cuda::std::make_unsigned_t<_Prom>;
  auto __a1     = static_cast<_UProm>(__a);
  auto __b1     = static_cast<_UProm>(__b);
  if constexpr (::cuda::std::is_signed_v<_Prom>)
  {
    return static_cast<_Common>((__a1 + __b1 - 1) / __b1);
  }
  else
  {
    if (::cuda::std::is_constant_evaluated())
    {
      const auto __res = __a1 / __b1;
      return static_cast<_Common>(__res + (__res * __b1 != __a1));
    }
    else
    {
      // the ::min method is faster even if __b is a compile-time constant
      NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                        (return static_cast<_Common>(::cuda::std::min(__a1, 1 + ((__a1 - 1) / __b1)));),
                        (const auto __res = __a1 / __b1; //
                         return static_cast<_Common>(__res + (__res * __b1 != __a1));))
    }
  }
}

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder, \p __b is an enum
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND ::cuda::std::is_enum_v<_Up>)
[[nodiscard]] _CCCL_API constexpr ::cuda::std::common_type_t<_Tp, ::cuda::std::underlying_type_t<_Up>>
ceil_div(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::ceil_div(__a, ::cuda::std::to_underlying(__b));
}

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder, \p __b is an enum
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(::cuda::std::is_enum_v<_Tp> _CCCL_AND ::cuda::std::is_integral_v<_Up>)
[[nodiscard]] _CCCL_API constexpr ::cuda::std::common_type_t<::cuda::std::underlying_type_t<_Tp>, _Up>
ceil_div(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::ceil_div(::cuda::std::to_underlying(__a), __b);
}

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder, \p __b is an enum
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(::cuda::std::is_enum_v<_Tp> _CCCL_AND ::cuda::std::is_enum_v<_Up>)
[[nodiscard]]
_CCCL_API constexpr ::cuda::std::common_type_t<::cuda::std::underlying_type_t<_Tp>, ::cuda::std::underlying_type_t<_Up>>
ceil_div(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::ceil_div(::cuda::std::to_underlying(__a), ::cuda::std::to_underlying(__b));
}

_CCCL_END_NAMESPACE_CUDA

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_CEIL_DIV_H
