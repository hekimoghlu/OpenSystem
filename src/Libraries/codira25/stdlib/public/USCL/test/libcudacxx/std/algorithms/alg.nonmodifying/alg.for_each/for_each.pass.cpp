/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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

// template<InputIterator Iter, Callable<auto, Iter::reference> Function>
//   requires CopyConstructible<Function>
//   constexpr Function   // constexpr after C++17
//   for_each(Iter first, Iter last, Function f);

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
  int ia[]       = {1, 3, 6, 7};
  int expected[] = {3, 5, 8, 9};

  cuda::std::for_each(cuda::std::begin(ia), cuda::std::end(ia), add_two{});
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
  {
    int ia[]         = {0, 1, 2, 3, 4, 5};
    const unsigned s = sizeof(ia) / sizeof(ia[0]);
    for_each_test f =
      cuda::std::for_each(cpp17_input_iterator<int*>(ia), cpp17_input_iterator<int*>(ia + s), for_each_test(0));
    assert(f.count == s);
    for (unsigned i = 0; i < s; ++i)
    {
      assert(ia[i] == static_cast<int>(i + 1));
    }
  }

  static_assert(test_constexpr(), "");

  return 0;
}
