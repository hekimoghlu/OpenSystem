/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_OVERFLOW_CAST_H
#define _CUDA___NUMERIC_OVERFLOW_CAST_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/__numeric/overflow_result.h>
#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__limits/numeric_limits.h>
#include <uscl/std/__type_traits/is_integer.h>
#include <uscl/std/__type_traits/is_signed.h>
#include <uscl/std/__utility/cmp.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _From, typename _To>
inline constexpr bool __is_integer_representable_v =
  ::cuda::std::cmp_less(::cuda::std::numeric_limits<_From>::max(), ::cuda::std::numeric_limits<_To>::max())
  && ::cuda::std::cmp_greater(::cuda::std::numeric_limits<_From>::min(), ::cuda::std::numeric_limits<_To>::min());

//! @brief Casts a number \p __from to a number of type \p _To with overflow detection
//! @param __from The number to cast
//! @return An overflow_result object containing the casted number and a boolean indicating whether an overflow
//! occurred
_CCCL_TEMPLATE(class _To, class _From)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_To> _CCCL_AND ::cuda::std::__cccl_is_cv_integer_v<_From>)
[[nodiscard]] _CCCL_API constexpr overflow_result<_To> overflow_cast(const _From& __from) noexcept
{
  bool __overflow = false;
  if constexpr (!__is_integer_representable_v<_From, _To>)
  {
    __overflow = !::cuda::std::in_range<_To>(__from);
  }
  return overflow_result<_To>{static_cast<_To>(__from), __overflow};
}

_CCCL_END_NAMESPACE_CUDA

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_OVERFLOW_CAST_H
