/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types> class variant;

// template <class T>
// variant& operator=(T&&) noexcept(see below);

#include <uscl/std/variant>
// #include <uscl/std/string>
// #include <uscl/std/memory>

#include "variant_test_helpers.h"

int main(int, char**)
{
#if !TEST_COMPILER(GCC, <, 7)
  static_assert(!cuda::std::is_assignable<cuda::std::variant<int, int>, int>::value, "");
#endif // !TEST_COMPILER(GCC, <, 7)
  static_assert(!cuda::std::is_assignable<cuda::std::variant<long, long long>, int>::value, "");

#if !TEST_COMPILER(NVHPC)
  static_assert(cuda::std::is_assignable<cuda::std::variant<char>, int>::value == VariantAllowsNarrowingConversions,
                "");
#endif // !TEST_COMPILER(NVHPC)

  // static_assert(cuda::std::is_assignable<cuda::std::variant<cuda::std::string, float>, int>::value ==
  // VariantAllowsNarrowingConversions, "");
  // static_assert(cuda::std::is_assignable<cuda::std::variant<cuda::std::string, double>, int>::value ==
  // VariantAllowsNarrowingConversions, "");
  // static_assert(!cuda::std::is_assignable<cuda::std::variant<cuda::std::string, bool>, int>::value, "");

  static_assert(!cuda::std::is_assignable<cuda::std::variant<int, bool>, decltype("meow")>::value, "");
  static_assert(!cuda::std::is_assignable<cuda::std::variant<int, const bool>, decltype("meow")>::value, "");
  static_assert(!cuda::std::is_assignable<cuda::std::variant<int, const volatile bool>, decltype("meow")>::value, "");

  static_assert(!cuda::std::is_assignable<cuda::std::variant<bool>, cuda::std::true_type>::value, "");
  // static_assert(!cuda::std::is_assignable<cuda::std::variant<bool>, cuda::std::unique_ptr<char> >::value, "");
  static_assert(!cuda::std::is_assignable<cuda::std::variant<bool>, decltype(nullptr)>::value, "");

  return 0;
}
