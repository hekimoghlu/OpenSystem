/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

// template<class T2, class E2> requires (!is_void_v<T2>)
//   friend constexpr bool operator==(const expected& x, const expected<T2, E2>& y);

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

// Test constraint
template <class T1, class T2, class = void>
constexpr bool CanCompare = false;

template <class T1, class T2>
constexpr bool CanCompare<T1, T2, cuda::std::void_t<decltype(cuda::std::declval<T1>() == cuda::std::declval<T2>())>> =
  true;

struct Foo
{};
static_assert(!CanCompare<Foo, Foo>, "");

static_assert(CanCompare<cuda::std::expected<int, int>, cuda::std::expected<int, int>>, "");
static_assert(CanCompare<cuda::std::expected<int, int>, cuda::std::expected<short, short>>, "");

// Note this is true because other overloads are unconstrained
static_assert(CanCompare<cuda::std::expected<int, int>, cuda::std::expected<void, int>>, "");

__host__ __device__ constexpr bool test()
{
  // x.has_value() && y.has_value()
  {
    const cuda::std::expected<int, int> e1(5);
    const cuda::std::expected<int, int> e2(10);
    const cuda::std::expected<int, int> e3(5);
    assert(e1 != e2);
    assert(e1 == e3);
  }

  // !x.has_value() && y.has_value()
  {
    const cuda::std::expected<int, int> e1(cuda::std::unexpect, 5);
    const cuda::std::expected<int, int> e2(10);
    const cuda::std::expected<int, int> e3(5);
    assert(e1 != e2);
    assert(e1 != e3);
  }

  // x.has_value() && !y.has_value()
  {
    const cuda::std::expected<int, int> e1(5);
    const cuda::std::expected<int, int> e2(cuda::std::unexpect, 10);
    const cuda::std::expected<int, int> e3(cuda::std::unexpect, 5);
    assert(e1 != e2);
    assert(e1 != e3);
  }

  // !x.has_value() && !y.has_value()
  {
    const cuda::std::expected<int, int> e1(cuda::std::unexpect, 5);
    const cuda::std::expected<int, int> e2(cuda::std::unexpect, 10);
    const cuda::std::expected<int, int> e3(cuda::std::unexpect, 5);
    assert(e1 != e2);
    assert(e1 == e3);
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
