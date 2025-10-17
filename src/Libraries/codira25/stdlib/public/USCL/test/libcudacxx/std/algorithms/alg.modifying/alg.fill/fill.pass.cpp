/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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

// template<ForwardIterator Iter, class T>
//   requires OutputIterator<Iter, const T&>
//   constexpr void      // constexpr after C++17
//   fill(Iter first, Iter last, const T& value);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
__host__ __device__ constexpr void test_char()
{
  const unsigned n = 4;
  char ca[n]       = {0};
  cuda::std::fill(Iter(ca), Iter(ca + n), char(1));
  assert(ca[0] == 1);
  assert(ca[1] == 1);
  assert(ca[2] == 1);
  assert(ca[3] == 1);
}

template <class Iter>
__host__ __device__ constexpr void test_int()
{
  const unsigned n = 4;
  int ia[n]        = {0};
  cuda::std::fill(Iter(ia), Iter(ia + n), 1);
  assert(ia[0] == 1);
  assert(ia[1] == 1);
  assert(ia[2] == 1);
  assert(ia[3] == 1);
}

__host__ __device__ constexpr bool test()
{
  test_char<forward_iterator<char*>>();
  test_char<bidirectional_iterator<char*>>();
  test_char<random_access_iterator<char*>>();
  test_char<char*>();

  test_int<forward_iterator<int*>>();
  test_int<bidirectional_iterator<int*>>();
  test_int<random_access_iterator<int*>>();
  test_int<int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
