/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class U, class... Args>
//   constexpr T& emplace(initializer_list<U> il, Args&&... args) noexcept;
// Constraints: is_nothrow_constructible_v<T, initializer_list<U>&, Args...> is true.
//
// Effects: Equivalent to:
// if (has_value()) {
//   destroy_at(addressof(val));
// } else {
//   destroy_at(addressof(unex));
//   has_val = true;
// }
// return *construct_at(addressof(val), il, cuda::std::forward<Args>(args)...);

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/span>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "../../types.h"
#include "test_macros.h"

template <class T, class... Args>
_CCCL_CONCEPT CanEmplace =
  _CCCL_REQUIRES_EXPR((T, variadic Args), T t, Args&&... args)((t.emplace(cuda::std::forward<Args>(args)...)));

static_assert(CanEmplace<cuda::std::expected<int, int>, int>, "");

template <bool Noexcept>
struct CtorFromInitalizerList
{
  __host__ __device__ CtorFromInitalizerList(cuda::std::initializer_list<int>&) noexcept(Noexcept);
  __host__ __device__ CtorFromInitalizerList(cuda::std::initializer_list<int>&, int) noexcept(Noexcept);
};

static_assert(CanEmplace<cuda::std::expected<CtorFromInitalizerList<true>, int>, cuda::std::initializer_list<int>&>,
              "");
static_assert(!CanEmplace<cuda::std::expected<CtorFromInitalizerList<false>, int>, cuda::std::initializer_list<int>&>,
              "");
static_assert(
  CanEmplace<cuda::std::expected<CtorFromInitalizerList<true>, int>, cuda::std::initializer_list<int>&, int>, "");
static_assert(
  !CanEmplace<cuda::std::expected<CtorFromInitalizerList<false>, int>, cuda::std::initializer_list<int>&, int>, "");

struct Data
{
  cuda::std::initializer_list<int> il;
  int i;

  __host__ __device__ constexpr Data(cuda::std::initializer_list<int>& l, int ii) noexcept
      : il(l)
      , i(ii)
  {}
};

__host__ __device__ constexpr bool
equal(const cuda::std::initializer_list<int>& lhs, const cuda::std::initializer_list<int>& rhs)
{
  auto* left  = lhs.begin();
  auto* right = rhs.begin();

  for (; left != rhs.end(); ++left, ++right)
  {
    assert(*left == *right);
  }
  assert(right == rhs.end());
  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // has_value
  {
    auto list1 = {1, 2, 3};
    auto list2 = {4, 5, 6};
    cuda::std::expected<Data, int> e(cuda::std::in_place, list1, 5);
    decltype(auto) x = e.emplace(list2, 10);
    static_assert(cuda::std::same_as<decltype(x), Data&>, "");
    assert(&x == &(*e));

    assert(e.has_value());
    assert(equal(e.value().il, list2));
    assert(e.value().i == 10);
  }

  // !has_value
  {
    auto list = {4, 5, 6};
    cuda::std::expected<Data, int> e(cuda::std::unexpect, 5);
    decltype(auto) x = e.emplace(list, 10);
    static_assert(cuda::std::same_as<decltype(x), Data&>, "");
    assert(&x == &(*e));

    assert(e.has_value());
    assert(equal(e.value().il, list));
    assert(e.value().i == 10);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
