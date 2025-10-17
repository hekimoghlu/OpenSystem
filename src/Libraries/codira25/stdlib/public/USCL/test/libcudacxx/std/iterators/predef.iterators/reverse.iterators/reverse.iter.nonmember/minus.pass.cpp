/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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

// <cuda/std/iterator>

// reverse_iterator

// template <RandomAccessIterator Iter1, RandomAccessIterator Iter2>
//   requires HasMinus<Iter2, Iter1>
// auto operator-(const reverse_iterator<Iter1>& x, const reverse_iterator<Iter2>& y) // constexpr in C++17
//  -> decltype(y.base() - x.base());

#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/iterator>
#include <uscl/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <class R1, class R2>
_CCCL_CONCEPT HasMinus = _CCCL_REQUIRES_EXPR((R1, R2), R1 r1, R2 r2)(unused(R1() - R2()));

template <class It1, class It2>
__host__ __device__ constexpr void test(It1 l, It2 r, cuda::std::ptrdiff_t x)
{
  const cuda::std::reverse_iterator<It1> r1(l);
  const cuda::std::reverse_iterator<It2> r2(r);
  assert((r1 - r2) == x);
}

__host__ __device__ constexpr bool tests()
{
  using PC  = const char*;
  char s[3] = {0};

  // Test same base iterator type
  test(s, s, 0);
  test(s, s + 1, 1);
  test(s + 1, s, -1);

  // Test different (but subtractable) base iterator types
  test(PC(s), s, 0);
  test(PC(s), s + 1, 1);
  test(PC(s + 1), s, -1);

  // Test non-subtractable base iterator types
  static_assert(HasMinus<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<int*>>, "");
  static_assert(HasMinus<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<const int*>>, "");
  static_assert(!HasMinus<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<char*>>, "");
  static_assert(!HasMinus<cuda::std::reverse_iterator<bidirectional_iterator<int*>>,
                          cuda::std::reverse_iterator<bidirectional_iterator<int*>>>,
                "");

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
