/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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

// template<ForwardIterator Iter>
//   requires EqualityComparable<Iter::value_type>
//   constexpr Iter  // constexpr after C++17
//   adjacent_find(Iter first, Iter last);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int ia[]          = {0, 1, 2, 2, 0, 1, 2, 3};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  assert(cuda::std::adjacent_find(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa))
         == forward_iterator<const int*>(ia + 2));
  assert(cuda::std::adjacent_find(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia))
         == forward_iterator<const int*>(ia));
  assert(cuda::std::adjacent_find(forward_iterator<const int*>(ia + 3), forward_iterator<const int*>(ia + sa))
         == forward_iterator<const int*>(ia + sa));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
