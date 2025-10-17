/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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

// template <class _Tp>
// inline constexpr empty_view<_Tp> empty{};

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void testType()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::views::empty<T>), const cuda::std::ranges::empty_view<T>>);
  static_assert(cuda::std::is_same_v<decltype((cuda::std::views::empty<T>) ), const cuda::std::ranges::empty_view<T>&>);

  auto v = cuda::std::views::empty<T>;
  assert(cuda::std::ranges::empty(v));
}

struct Empty
{};
struct BigType
{
  char buff[8];
};

__host__ __device__ constexpr bool test()
{
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
