/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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

// UNSUPPORTED: msvc-19.16

// template<class T>
// class empty_view;

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void testType()
{
  static_assert(cuda::std::ranges::range<cuda::std::ranges::empty_view<T>>);
  static_assert(cuda::std::ranges::range<const cuda::std::ranges::empty_view<T>>);
  static_assert(cuda::std::ranges::view<cuda::std::ranges::empty_view<T>>);

  cuda::std::ranges::empty_view<T> empty{};

  assert(empty.begin() == nullptr);
  assert(empty.end() == nullptr);
  assert(empty.data() == nullptr);
  assert(empty.size() == 0);
  assert(empty.empty() == true);

  assert(cuda::std::ranges::begin(empty) == nullptr);
  assert(cuda::std::ranges::end(empty) == nullptr);
  assert(cuda::std::ranges::data(empty) == nullptr);
  assert(cuda::std::ranges::size(empty) == 0);
  assert(cuda::std::ranges::empty(empty) == true);
}

struct Empty
{};
struct BigType
{
  char buff[8];
};

#if TEST_STD_VER >= 2020
template <class T>
concept ValidEmptyView = requires { typename cuda::std::ranges::empty_view<T>; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
constexpr bool ValidEmptyView = false;

template <class T>
constexpr bool ValidEmptyView<T, cuda::std::void_t<cuda::std::ranges::empty_view<T>>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  // Not objects:
  static_assert(!ValidEmptyView<int&>);
  static_assert(!ValidEmptyView<void>);

  testType<int>();
  testType<const int>();
  testType<int*>();
  testType<Empty>();
  testType<const Empty>();
  testType<BigType>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
