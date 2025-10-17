/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

// <algorithm>

// template<ForwardIterator Iter, Predicate<auto, Iter::value_type> Pred, class T>
//   requires OutputIterator<Iter, Iter::reference>
//         && OutputIterator<Iter, const T&>
//         && CopyConstructible<Pred>
//   constexpr void      // constexpr after C++17
//   replace_if(Iter first, Iter last, Pred pred, const T& new_value);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

constexpr __host__ __device__ bool equalToTwo(const int v) noexcept
{
  return v == 2;
}

template <class Iter>
constexpr __host__ __device__ void test()
{
  constexpr int N           = 5;
  int ia[N]                 = {0, 1, 2, 3, 4};
  constexpr int expected[N] = {0, 1, 5, 3, 4};
  cuda::std::replace_if(Iter(ia), Iter(ia + N), equalToTwo, 5);

  for (int i = 0; i < N; ++i)
  {
    assert(ia[i] == expected[i]);
  }
}

constexpr __host__ __device__ bool test()
{
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
