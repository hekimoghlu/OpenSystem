/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

#ifndef _CUDA_STD___RANDOM_IS_VALID_H
#define _CUDA_STD___RANDOM_IS_VALID_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/integral_constant.h>
#include <uscl/std/__type_traits/is_same.h>
#include <uscl/std/__type_traits/is_unsigned.h>
#include <uscl/std/__utility/declval.h>
#include <uscl/std/cstdint>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// [rand.req.genl]/1.4:
// The effect of instantiating a template that has a template type parameter
// named RealType is undefined unless the corresponding template argument is
// cv-unqualified and is one of float, double, or long double.

template <class>
inline constexpr bool __libcpp_random_is_valid_realtype = false;

template <>
inline constexpr bool __libcpp_random_is_valid_realtype<float> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_realtype<double> = true;
#if _CCCL_HAS_LONG_DOUBLE()
template <>
inline constexpr bool __libcpp_random_is_valid_realtype<long double> = true;
#endif // _CCCL_HAS_LONG_DOUBLE()

// [rand.req.genl]/1.5:
// The effect of instantiating a template that has a template type parameter
// named IntType is undefined unless the corresponding template argument is
// cv-unqualified and is one of short, int, long, long long, unsigned short,
// unsigned int, unsigned long, or unsigned long long.

template <class>
inline constexpr bool __libcpp_random_is_valid_inttype = false;
template <> // extension
inline constexpr bool __libcpp_random_is_valid_inttype<int8_t> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<short> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<int> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<long> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<long long> = true;
template <> // extension
inline constexpr bool __libcpp_random_is_valid_inttype<uint8_t> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<unsigned short> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<unsigned int> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<unsigned long> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<unsigned long long> = true;
#if _CCCL_HAS_INT128()
template <> // extension
inline constexpr bool __libcpp_random_is_valid_inttype<__int128_t> = true;
template <> // extension
inline constexpr bool __libcpp_random_is_valid_inttype<__uint128_t> = true;
#endif // _CCCL_HAS_INT128()

// [rand.req.urng]/3:
// A class G meets the uniform random bit generator requirements if G models
// uniform_random_bit_generator, invoke_result_t<G&> is an unsigned integer type,
// and G provides a nested typedef-name result_type that denotes the same type
// as invoke_result_t<G&>.
// (In particular, reject URNGs with signed result_types; our distributions cannot
// handle such generator types.)

template <class, class = void>
inline constexpr bool __cccl_random_is_valid_urng = false;
template <class _Gp>
inline constexpr bool __cccl_random_is_valid_urng<
  _Gp,
  enable_if_t<is_unsigned_v<typename _Gp::result_type>
              && is_same_v<decltype(::cuda::std::declval<_Gp&>()()), typename _Gp::result_type>>> = true;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_IS_VALID_H
