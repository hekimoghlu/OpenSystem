/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
// concept floating_point = // see below

#include <uscl/std/concepts>
#include <uscl/std/type_traits>

#include "arithmetic.h"
#include "test_macros.h"

using cuda::std::floating_point;

template <typename T>
__host__ __device__ constexpr bool CheckFloatingPointQualifiers()
{
  constexpr bool result = floating_point<T>;
  static_assert(floating_point<const T> == result, "");
  static_assert(floating_point<volatile T> == result, "");
  static_assert(floating_point<const volatile T> == result, "");

  static_assert(!floating_point<T&>, "");
  static_assert(!floating_point<const T&>, "");
  static_assert(!floating_point<volatile T&>, "");
  static_assert(!floating_point<const volatile T&>, "");

  static_assert(!floating_point<T&&>, "");
  static_assert(!floating_point<const T&&>, "");
  static_assert(!floating_point<volatile T&&>, "");
  static_assert(!floating_point<const volatile T&&>, "");

  static_assert(!floating_point<T*>, "");
  static_assert(!floating_point<const T*>, "");
  static_assert(!floating_point<volatile T*>, "");
  static_assert(!floating_point<const volatile T*>, "");

  static_assert(!floating_point<T (*)()>, "");
  static_assert(!floating_point<T (&)()>, "");
  static_assert(!floating_point<T (&&)()>, "");

  return result;
}

// floating-point types
static_assert(CheckFloatingPointQualifiers<float>(), "");
static_assert(CheckFloatingPointQualifiers<double>(), "");
static_assert(CheckFloatingPointQualifiers<long double>(), "");

// types that aren't floating-point
static_assert(!CheckFloatingPointQualifiers<signed char>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned char>(), "");
static_assert(!CheckFloatingPointQualifiers<short>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned short>(), "");
static_assert(!CheckFloatingPointQualifiers<int>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned int>(), "");
static_assert(!CheckFloatingPointQualifiers<long>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned long>(), "");
static_assert(!CheckFloatingPointQualifiers<long long>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned long long>(), "");
static_assert(!CheckFloatingPointQualifiers<wchar_t>(), "");
static_assert(!CheckFloatingPointQualifiers<bool>(), "");
static_assert(!CheckFloatingPointQualifiers<char>(), "");
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(!CheckFloatingPointQualifiers<char8_t>(), "");
#endif // TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(!CheckFloatingPointQualifiers<char16_t>(), "");
static_assert(!CheckFloatingPointQualifiers<char32_t>(), "");
static_assert(!floating_point<void>, "");

static_assert(!CheckFloatingPointQualifiers<ClassicEnum>(), "");
static_assert(!CheckFloatingPointQualifiers<ScopedEnum>(), "");
static_assert(!CheckFloatingPointQualifiers<EmptyStruct>(), "");
static_assert(!CheckFloatingPointQualifiers<int EmptyStruct::*>(), "");
static_assert(!CheckFloatingPointQualifiers<int (EmptyStruct::*)()>(), "");

int main(int, char**)
{
  return 0;
}
