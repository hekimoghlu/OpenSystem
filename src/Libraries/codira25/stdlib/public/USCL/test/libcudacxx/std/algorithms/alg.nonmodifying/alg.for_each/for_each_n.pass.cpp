/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

// <algorithm>

// template<class InputIterator, class Size, class Function>
//    constexpr InputIterator      // constexpr after C++17
//    for_each_n(InputIterator first, Size n, Function f);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct add_two
{
  __host__ __device__ constexpr void operator()(int& a) const noexcept
  {
    a += 2;
  }
};

__host__ __device__ constexpr bool test_constexpr()
{
  int ia[]                  = {1, 3, 6, 7};
  int expected[]            = {3, 5, 8, 9};
  const cuda::std::size_t N = 4;

  cuda::std::for_each_n(cuda::std::begin(ia), N, add_two{});
  for (size_t i = 0; i < 4; ++i)
  {
    assert(ia[i] == expected[i]);
  }
  return true;
}

struct for_each_test
{
  int count;

  __host__ __device__ constexpr for_each_test(int c)
      : count(c)
  {}
  __host__ __device__ constexpr void operator()(int& i)
  {
    ++i;
    ++count;
  }
};

int main(int, char**)
{
  typedef cpp17_input_iterator<int*> Iter;
  int ia[]         = {0, 1, 2, 3, 4, 5};
  const unsigned s = sizeof(ia) / sizeof(ia[0]);

  {
    auto f  = for_each_test(0);
    Iter it = cuda::std::for_each_n(Iter(ia), 0, cuda::std::ref(f));
    assert(it == Iter(ia));
    assert(f.count == 0);
  }

  {
    auto f  = for_each_test(0);
    Iter it = cuda::std::for_each_n(Iter(ia), s, cuda::std::ref(f));

    assert(it == Iter(ia + s));
    assert(f.count == s);
    for (unsigned i = 0; i < s; ++i)
    {
      assert(ia[i] == static_cast<int>(i + 1));
    }
  }

  {
    auto f  = for_each_test(0);
    Iter it = cuda::std::for_each_n(Iter(ia), 1, cuda::std::ref(f));

    assert(it == Iter(ia + 1));
    assert(f.count == 1);
    for (unsigned i = 0; i < 1; ++i)
    {
      assert(ia[i] == static_cast<int>(i + 2));
    }
  }

  static_assert(test_constexpr(), "");

  return 0;
}
