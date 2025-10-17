/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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

// requires RandomAccessIterator<Iter>
// reverse_iterator& operator-=(difference_type n); // constexpr in C++17

#include <uscl/std/cassert>
#include <uscl/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ constexpr void test(It i, typename cuda::std::iterator_traits<It>::difference_type n, It x)
{
  cuda::std::reverse_iterator<It> r(i);
  cuda::std::reverse_iterator<It>& rr = r -= n;
  assert(r.base() == x);
  assert(&rr == &r);
}

__host__ __device__ constexpr bool tests()
{
  const char* s = "1234567890";
  test(random_access_iterator<const char*>(s + 5), 5, random_access_iterator<const char*>(s + 10));
  test(s + 5, 5, s + 10);
  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
