/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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

// template <size_t I, class U, class ...Args>
//   variant_alternative_t<I, variant<Types...>>& emplace(initializer_list<U> il,Args&&... args);

#include <uscl/std/cassert>
// #include <uscl/std/string>
#include <uscl/std/type_traits>
#include <uscl/std/variant>

#include "archetypes.h"
#include "test_convertible.h"
#include "test_macros.h"

struct InitList
{
  cuda::std::size_t size;
  __host__ __device__ constexpr InitList(cuda::std::initializer_list<int> il)
      : size(il.size())
  {}
};

struct InitListArg
{
  cuda::std::size_t size;
  int value;
  __host__ __device__ constexpr InitListArg(cuda::std::initializer_list<int> il, int v)
      : size(il.size())
      , value(v)
  {}
};

template <class Var, size_t I, class... Args>
__host__ __device__ constexpr auto test_emplace_exists_imp(int)
  -> decltype(cuda::std::declval<Var>().template emplace<I>(cuda::std::declval<Args>()...), true)
{
  return true;
}

template <class, size_t, class...>
__host__ __device__ constexpr auto test_emplace_exists_imp(long) -> bool
{
  return false;
}

template <class Var, size_t I, class... Args>
__host__ __device__ constexpr bool emplace_exists()
{
  return test_emplace_exists_imp<Var, I, Args...>(0);
}

__host__ __device__ void test_emplace_sfinae()
{
  using V  = cuda::std::variant<int, TestTypes::NoCtors, InitList, InitListArg, long, long>;
  using IL = cuda::std::initializer_list<int>;
  static_assert(!emplace_exists<V, 1, IL>(), "no such constructor");
  static_assert(emplace_exists<V, 2, IL>(), "");
  static_assert(!emplace_exists<V, 2, int>(), "args don't match");
  static_assert(!emplace_exists<V, 2, IL, int>(), "too many args");
  static_assert(emplace_exists<V, 3, IL, int>(), "");
  static_assert(!emplace_exists<V, 3, int>(), "args don't match");
  static_assert(!emplace_exists<V, 3, IL>(), "too few args");
  static_assert(!emplace_exists<V, 3, IL, int, int>(), "too many args");
}

__host__ __device__ void test_basic()
{
  using V = cuda::std::variant<int, InitList, InitListArg, TestTypes::NoCtors>;
  V v;
  auto& ref1 = v.emplace<1>({1, 2, 3});
  static_assert(cuda::std::is_same_v<InitList&, decltype(ref1)>, "");
  assert(cuda::std::get<1>(v).size == 3);
  assert(&ref1 == &cuda::std::get<1>(v));
  auto& ref2 = v.emplace<2>({1, 2, 3, 4}, 42);
  static_assert(cuda::std::is_same_v<InitListArg&, decltype(ref2)>, "");
  assert(cuda::std::get<2>(v).size == 4);
  assert(cuda::std::get<2>(v).value == 42);
  assert(&ref2 == &cuda::std::get<2>(v));
  auto& ref3 = v.emplace<1>({1});
  static_assert(cuda::std::is_same_v<InitList&, decltype(ref3)>, "");
  assert(cuda::std::get<1>(v).size == 1);
  assert(&ref3 == &cuda::std::get<1>(v));
}

int main(int, char**)
{
  test_basic();
  test_emplace_sfinae();

  return 0;
}
