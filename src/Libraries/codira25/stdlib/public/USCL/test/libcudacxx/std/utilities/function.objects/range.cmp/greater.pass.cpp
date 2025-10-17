/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// ranges::greater

#include <uscl/std/cassert>
#include <uscl/std/functional>
#include <uscl/std/type_traits>

#include "compare_types.h"
#include "MoveOnly.h"
#include "pointer_comparison_test_helper.h"
#include "test_macros.h"

struct NotTotallyOrdered
{
  __host__ __device__ friend bool operator<(const NotTotallyOrdered&, const NotTotallyOrdered&);
};

static_assert(!cuda::std::is_invocable_v<cuda::std::ranges::greater, NotTotallyOrdered, NotTotallyOrdered>);
#if !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017 // MSVC considers implicit conversions in C++17
static_assert(!cuda::std::is_invocable_v<cuda::std::ranges::greater, int, MoveOnly>);
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017
static_assert(cuda::std::is_invocable_v<cuda::std::ranges::greater, explicit_operators, explicit_operators>);

#if TEST_STD_VER > 2017
static_assert(requires { typename cuda::std::ranges::greater::is_transparent; });
#else
template <class T, class = void>
inline constexpr bool is_transparent = false;
template <class T>
inline constexpr bool is_transparent<T, cuda::std::void_t<typename T::is_transparent>> = true;
static_assert(is_transparent<cuda::std::ranges::greater>);
#endif

__host__ __device__ constexpr bool test()
{
  auto fn = cuda::std::ranges::greater();

  assert(fn(MoveOnly(42), MoveOnly(41)));

  ForwardingTestObject a{};
  ForwardingTestObject b{};
  assert(!fn(a, b));
  assert(fn(cuda::std::move(a), cuda::std::move(b)));

  assert(!fn(2, 2));
  assert(!fn(1, 2));
  assert(fn(2, 1));

  assert(fn(2, 1L));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  // test total ordering of int* for greater<int*> and greater<void>.
  do_pointer_comparison_test(cuda::std::ranges::greater());

  return 0;
}
