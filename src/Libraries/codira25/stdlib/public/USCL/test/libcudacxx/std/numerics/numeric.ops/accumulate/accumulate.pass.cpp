/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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

// <numeric>

// Became constexpr in C++20
// template <InputIterator Iter, MoveConstructible T>
//   requires HasPlus<T, Iter::reference>
//         && HasAssign<T, HasPlus<T, Iter::reference>::result_type>
//   T
//   accumulate(Iter first, Iter last, T init);

#include <uscl/std/cassert>
#include <uscl/std/numeric>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class T>
__host__ __device__ constexpr void test(Iter first, Iter last, T init, T x)
{
  assert(cuda::std::accumulate(first, last, init) == x);
}

template <class Iter>
__host__ __device__ constexpr void test()
{
  int ia[]    = {1, 2, 3, 4, 5, 6};
  unsigned sa = sizeof(ia) / sizeof(ia[0]);
  test(Iter(ia), Iter(ia), 0, 0);
  test(Iter(ia), Iter(ia), 10, 10);
  test(Iter(ia), Iter(ia + 1), 0, 1);
  test(Iter(ia), Iter(ia + 1), 10, 11);
  test(Iter(ia), Iter(ia + 2), 0, 3);
  test(Iter(ia), Iter(ia + 2), 10, 13);
  test(Iter(ia), Iter(ia + sa), 0, 21);
  test(Iter(ia), Iter(ia + sa), 10, 31);
}

__host__ __device__ constexpr bool test()
{
  test<cpp17_input_iterator<const int*>>();
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
