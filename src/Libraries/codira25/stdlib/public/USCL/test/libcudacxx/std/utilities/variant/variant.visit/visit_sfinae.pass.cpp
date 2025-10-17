/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
// template <class Visitor, class... Variants>
// constexpr see below visit(Visitor&& vis, Variants&&... vars);

#include <uscl/std/cassert>
// #include <uscl/std/memory>
// #include <uscl/std/string>
#include <uscl/std/type_traits>
#include <uscl/std/utility>
#include <uscl/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct any_visitor
{
  template <typename T>
  __host__ __device__ void operator()(const T&) const
  {}
};

template <typename T, typename = decltype(cuda::std::visit(cuda::std::declval<any_visitor&>(), cuda::std::declval<T>()))>
__host__ __device__ constexpr bool has_visit(int)
{
  return true;
}

template <typename T>
__host__ __device__ constexpr bool has_visit(...)
{
  return false;
}

__host__ __device__ void test_sfinae()
{
  struct BadVariant
      : cuda::std::variant<short>
      , cuda::std::variant<long, float>
  {};
  struct BadVariant2 : private cuda::std::variant<long, float>
  {};
  struct GoodVariant : cuda::std::variant<long, float>
  {};
  struct GoodVariant2 : GoodVariant
  {};

  static_assert(!has_visit<int>(0), "");
#if !TEST_COMPILER(MSVC) // MSVC cannot deal with that even with std::variant
  static_assert(!has_visit<BadVariant>(0), "");
#endif // !TEST_COMPILER(MSVC)
  static_assert(!has_visit<BadVariant2>(0), "");
  static_assert(has_visit<cuda::std::variant<int>>(0), "");
  static_assert(has_visit<GoodVariant>(0), "");
  static_assert(has_visit<GoodVariant2>(0), "");
}

int main(int, char**)
{
  test_sfinae();

  return 0;
}
