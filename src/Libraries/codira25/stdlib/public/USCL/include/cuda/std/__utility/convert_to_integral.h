/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_CONVERT_TO_INTEGRAL_H
#define _CUDA_STD___UTILITY_CONVERT_TO_INTEGRAL_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/is_enum.h>
#include <uscl/std/__type_traits/is_floating_point.h>
#include <uscl/std/__type_traits/underlying_type.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_API constexpr int __convert_to_integral(int __val)
{
  return __val;
}

_CCCL_API constexpr unsigned __convert_to_integral(unsigned __val)
{
  return __val;
}

_CCCL_API constexpr long __convert_to_integral(long __val)
{
  return __val;
}

_CCCL_API constexpr unsigned long __convert_to_integral(unsigned long __val)
{
  return __val;
}

_CCCL_API constexpr long long __convert_to_integral(long long __val)
{
  return __val;
}

_CCCL_API constexpr unsigned long long __convert_to_integral(unsigned long long __val)
{
  return __val;
}

template <typename _Fp>
_CCCL_API constexpr enable_if_t<is_floating_point<_Fp>::value, long long> __convert_to_integral(_Fp __val)
{
  return __val;
}

#if _CCCL_HAS_INT128()
_CCCL_API constexpr __int128_t __convert_to_integral(__int128_t __val)
{
  return __val;
}

_CCCL_API constexpr __uint128_t __convert_to_integral(__uint128_t __val)
{
  return __val;
}
#endif // _CCCL_HAS_INT128()

template <class _Tp, bool = is_enum<_Tp>::value>
struct __sfinae_underlying_type
{
  using type            = typename underlying_type<_Tp>::type;
  using __promoted_type = decltype(((type) 1) + 0);
};

template <class _Tp>
struct __sfinae_underlying_type<_Tp, false>
{};

template <class _Tp>
_CCCL_API constexpr typename __sfinae_underlying_type<_Tp>::__promoted_type __convert_to_integral(_Tp __val)
{
  return __val;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_CONVERT_TO_INTEGRAL_H
