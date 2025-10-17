/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
// template<class InputIterator, class T>
//   T reduce(InputIterator first, InputIterator last, T init);

#include <uscl/std/cassert>
#include <uscl/std/numeric>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class T>
__host__ __device__ constexpr void test(Iter first, Iter last, T init, T x)
{
  static_assert(cuda::std::is_same<T, decltype(cuda::std::reduce(first, last, init))>::value, "");
  assert(cuda::std::reduce(first, last, init) == x);
}

template <class Iter>
__host__ __device__ constexpr void test()
{
  int ia[]    = {1, 2, 3, 4, 5, 6};
  unsigned sa = sizeof(ia) / sizeof(ia[0]);
  test(Iter(ia), Iter(ia), 0, 0);
  test(Iter(ia), Iter(ia), 1, 1);
  test(Iter(ia), Iter(ia + 1), 0, 1);
  test(Iter(ia), Iter(ia + 1), 2, 3);
  test(Iter(ia), Iter(ia + 2), 0, 3);
  test(Iter(ia), Iter(ia + 2), 3, 6);
  test(Iter(ia), Iter(ia + sa), 0, 21);
  test(Iter(ia), Iter(ia + sa), 4, 25);
}

template <typename T, typename Init>
__host__ __device__ constexpr void test_return_type()
{
  T* p = nullptr;
  unused(p);
  static_assert(cuda::std::is_same<Init, decltype(cuda::std::reduce(p, p, Init{}))>::value, "");
}

__host__ __device__ constexpr bool test()
{
  test_return_type<char, int>();
  test_return_type<int, int>();
  test_return_type<int, unsigned long>();
  test_return_type<float, int>();
  test_return_type<short, float>();
  test_return_type<double, char>();
  test_return_type<char, double>();

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
