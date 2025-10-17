/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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

// template<class T>
// concept unsigned_integral = // see below

#include <uscl/std/concepts>
#include <uscl/std/type_traits>

#include "arithmetic.h"
#include "test_macros.h"

using cuda::std::unsigned_integral;

template <typename T>
__host__ __device__ constexpr bool CheckUnsignedIntegralQualifiers()
{
  constexpr bool result = unsigned_integral<T>;
  static_assert(unsigned_integral<const T> == result, "");
  static_assert(unsigned_integral<volatile T> == result, "");
  static_assert(unsigned_integral<const volatile T> == result, "");

  static_assert(!unsigned_integral<T&>, "");
  static_assert(!unsigned_integral<const T&>, "");
  static_assert(!unsigned_integral<volatile T&>, "");
  static_assert(!unsigned_integral<const volatile T&>, "");

  static_assert(!unsigned_integral<T&&>, "");
  static_assert(!unsigned_integral<const T&&>, "");
  static_assert(!unsigned_integral<volatile T&&>, "");
  static_assert(!unsigned_integral<const volatile T&&>, "");

  static_assert(!unsigned_integral<T*>, "");
  static_assert(!unsigned_integral<const T*>, "");
  static_assert(!unsigned_integral<volatile T*>, "");
  static_assert(!unsigned_integral<const volatile T*>, "");

  static_assert(!unsigned_integral<T (*)()>, "");
  static_assert(!unsigned_integral<T (&)()>, "");
  static_assert(!unsigned_integral<T (&&)()>, "");

  return result;
}

// standard unsigned types
static_assert(CheckUnsignedIntegralQualifiers<unsigned char>(), "");
static_assert(CheckUnsignedIntegralQualifiers<unsigned short>(), "");
static_assert(CheckUnsignedIntegralQualifiers<unsigned int>(), "");
static_assert(CheckUnsignedIntegralQualifiers<unsigned long>(), "");
static_assert(CheckUnsignedIntegralQualifiers<unsigned long long>(), "");

// Whether bool and character types are signed or unsigned is impl-defined
static_assert(CheckUnsignedIntegralQualifiers<wchar_t>() == !cuda::std::is_signed_v<wchar_t>, "");
static_assert(CheckUnsignedIntegralQualifiers<bool>() == !cuda::std::is_signed_v<bool>, "");
static_assert(CheckUnsignedIntegralQualifiers<char>() == !cuda::std::is_signed_v<char>, "");
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(CheckUnsignedIntegralQualifiers<char8_t>() == !cuda::std::is_signed_v<char8_t>, "");
#endif // TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(CheckUnsignedIntegralQualifiers<char16_t>() == !cuda::std::is_signed_v<char16_t>, "");
static_assert(CheckUnsignedIntegralQualifiers<char32_t>() == !cuda::std::is_signed_v<char32_t>, "");

// extended integers
#if _CCCL_HAS_INT128()
static_assert(CheckUnsignedIntegralQualifiers<__uint128_t>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<__int128_t>(), "");
#endif

// integer types that aren't unsigned integrals
static_assert(!CheckUnsignedIntegralQualifiers<signed char>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<short>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<int>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<long>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<long long>(), "");

static_assert(!unsigned_integral<void>, "");
static_assert(!CheckUnsignedIntegralQualifiers<float>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<double>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<long double>(), "");

static_assert(!CheckUnsignedIntegralQualifiers<ClassicEnum>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<ScopedEnum>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<EmptyStruct>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<int EmptyStruct::*>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<int (EmptyStruct::*)()>(), "");

#if TEST_STD_VER > 2017
static_assert(CheckSubsumption(0), "");
static_assert(CheckSubsumption(0U), "");
#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  return 0;
}
