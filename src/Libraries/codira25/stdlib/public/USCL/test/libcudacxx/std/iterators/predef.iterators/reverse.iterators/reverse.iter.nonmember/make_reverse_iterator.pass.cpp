/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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

// template <class Iterator>
// reverse_iterator<Iterator> make_reverse_iterator(Iterator i); // constexpr in C++17

#include <uscl/std/cassert>
#include <uscl/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ constexpr void test(It i)
{
  const cuda::std::reverse_iterator<It> r = cuda::std::make_reverse_iterator(i);
  assert(r.base() == i);
}

__host__ __device__ constexpr bool tests()
{
  const char* s = "1234567890";
  random_access_iterator<const char*> b(s);
  random_access_iterator<const char*> e(s + 10);
  while (b != e)
  {
    test(b++);
  }
  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
