/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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

// template <InputIterator Iter>
//   Iter next(Iter x, Iter::difference_type n = 1);

// LWG #2353 relaxed the requirement on next from ForwardIterator to InputIterator

#include <uscl/std/cassert>
#include <uscl/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ constexpr void
check_next_n(It it, typename cuda::std::iterator_traits<It>::difference_type n, It result)
{
  static_assert(cuda::std::is_same<decltype(cuda::std::next(it, n)), It>::value, "");
  assert(cuda::std::next(it, n) == result);

  It (*next_ptr)(It, typename cuda::std::iterator_traits<It>::difference_type) = cuda::std::next;
  assert(next_ptr(it, n) == result);
}

template <class It>
__host__ __device__ constexpr void check_next_1(It it, It result)
{
  static_assert(cuda::std::is_same<decltype(cuda::std::next(it)), It>::value, "");
  assert(cuda::std::next(it) == result);
}

__host__ __device__ constexpr bool tests()
{
  const char* s = "1234567890";
  check_next_n(cpp17_input_iterator<const char*>(s), 10, cpp17_input_iterator<const char*>(s + 10));
  check_next_n(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s + 10));
  check_next_n(bidirectional_iterator<const char*>(s), 10, bidirectional_iterator<const char*>(s + 10));
  check_next_n(bidirectional_iterator<const char*>(s + 10), -10, bidirectional_iterator<const char*>(s));
  check_next_n(random_access_iterator<const char*>(s), 10, random_access_iterator<const char*>(s + 10));
  check_next_n(random_access_iterator<const char*>(s + 10), -10, random_access_iterator<const char*>(s));
  check_next_n(s, 10, s + 10);

  check_next_1(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 1));
  check_next_1(forward_iterator<const char*>(s), forward_iterator<const char*>(s + 1));
  check_next_1(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s + 1));
  check_next_1(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s + 1));
  check_next_1(s, s + 1);

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
