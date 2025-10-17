/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
// concept integral = // see below

#include <uscl/std/concepts>
#include <uscl/std/type_traits>

#include "arithmetic.h"
#include "test_macros.h"

using cuda::std::integral;

template <typename T>
__host__ __device__ constexpr bool CheckIntegralQualifiers()
{
  constexpr bool result = integral<T>;
  static_assert(integral<const T> == result, "");
  static_assert(integral<volatile T> == result, "");
  static_assert(integral<const volatile T> == result, "");

  static_assert(!integral<T&>, "");
  static_assert(!integral<const T&>, "");
  static_assert(!integral<volatile T&>, "");
  static_assert(!integral<const volatile T&>, "");

  static_assert(!integral<T&&>, "");
  static_assert(!integral<const T&&>, "");
  static_assert(!integral<volatile T&&>, "");
  static_assert(!integral<const volatile T&&>, "");

  static_assert(!integral<T*>, "");
  static_assert(!integral<const T*>, "");
  static_assert(!integral<volatile T*>, "");
  static_assert(!integral<const volatile T*>, "");

  static_assert(!integral<T (*)()>, "");
  static_assert(!integral<T (&)()>, "");
  static_assert(!integral<T (&&)()>, "");

  return result;
}

// standard signed and unsigned integers
static_assert(CheckIntegralQualifiers<signed char>(), "");
static_assert(CheckIntegralQualifiers<unsigned char>(), "");
static_assert(CheckIntegralQualifiers<short>(), "");
static_assert(CheckIntegralQualifiers<unsigned short>(), "");
static_assert(CheckIntegralQualifiers<int>(), "");
static_assert(CheckIntegralQualifiers<unsigned int>(), "");
static_assert(CheckIntegralQualifiers<long>(), "");
static_assert(CheckIntegralQualifiers<unsigned long>(), "");
static_assert(CheckIntegralQualifiers<long long>(), "");
static_assert(CheckIntegralQualifiers<unsigned long long>(), "");

// extended integers
#if _CCCL_HAS_INT128()
static_assert(CheckIntegralQualifiers<__int128_t>(), "");
static_assert(CheckIntegralQualifiers<__uint128_t>(), "");
#endif

// bool and char types are also integral
static_assert(CheckIntegralQualifiers<wchar_t>(), "");
static_assert(CheckIntegralQualifiers<bool>(), "");
static_assert(CheckIntegralQualifiers<char>(), "");
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(CheckIntegralQualifiers<char8_t>(), "");
#endif // TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(CheckIntegralQualifiers<char16_t>(), "");
static_assert(CheckIntegralQualifiers<char32_t>(), "");

// types that aren't integral
static_assert(!integral<void>, "");
static_assert(!CheckIntegralQualifiers<float>(), "");
static_assert(!CheckIntegralQualifiers<double>(), "");
static_assert(!CheckIntegralQualifiers<long double>(), "");

static_assert(!CheckIntegralQualifiers<ClassicEnum>(), "");

static_assert(!CheckIntegralQualifiers<ScopedEnum>(), "");

static_assert(!CheckIntegralQualifiers<EmptyStruct>(), "");
static_assert(!CheckIntegralQualifiers<int EmptyStruct::*>(), "");
static_assert(!CheckIntegralQualifiers<int (EmptyStruct::*)()>(), "");

#if TEST_STD_VER > 2017
static_assert(CheckSubsumption(0), "");
static_assert(CheckSubsumption(0U), "");
#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  return 0;
}
