/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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

// template <BidirectionalIterator Iter1, BidirectionalIterator Iter2>
//   requires HasEqualTo<Iter1, Iter2>
// bool operator==(const reverse_iterator<Iter1>& x, const reverse_iterator<Iter2>& y); // constexpr since C++17

#include <uscl/std/cassert>
#include <uscl/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ constexpr void test(It l, It r, bool x)
{
  const cuda::std::reverse_iterator<It> r1(l);
  const cuda::std::reverse_iterator<It> r2(r);
  assert((r1 == r2) == x);
}

__host__ __device__ constexpr bool tests()
{
  const char* s = "1234567890";
  test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s), true);
  test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s + 1), false);
  test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s), true);
  test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s + 1), false);
  test(s, s, true);
  test(s, s + 1, false);
  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
