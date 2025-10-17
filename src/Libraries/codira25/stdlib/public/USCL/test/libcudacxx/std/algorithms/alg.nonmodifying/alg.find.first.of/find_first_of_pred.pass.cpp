/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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

// template<InputIterator Iter1, ForwardIterator Iter2,
//          Predicate<auto, Iter1::value_type, Iter2::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr Iter1  // constexpr after C++17
//   find_first_of(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Pred pred);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>
#include <uscl/std/functional>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int ia[]          = {0, 1, 2, 3, 0, 1, 2, 3};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  int ib[]          = {1, 3, 5, 7};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  assert(cuda::std::find_first_of(
           cpp17_input_iterator<const int*>(ia),
           cpp17_input_iterator<const int*>(ia + sa),
           forward_iterator<const int*>(ib),
           forward_iterator<const int*>(ib + sb),
           cuda::std::equal_to<int>())
         == cpp17_input_iterator<const int*>(ia + 1));
  int ic[] = {7};
  assert(cuda::std::find_first_of(
           cpp17_input_iterator<const int*>(ia),
           cpp17_input_iterator<const int*>(ia + sa),
           forward_iterator<const int*>(ic),
           forward_iterator<const int*>(ic + 1),
           cuda::std::equal_to<int>())
         == cpp17_input_iterator<const int*>(ia + sa));
  assert(cuda::std::find_first_of(
           cpp17_input_iterator<const int*>(ia),
           cpp17_input_iterator<const int*>(ia + sa),
           forward_iterator<const int*>(ic),
           forward_iterator<const int*>(ic),
           cuda::std::equal_to<int>())
         == cpp17_input_iterator<const int*>(ia + sa));
  assert(cuda::std::find_first_of(
           cpp17_input_iterator<const int*>(ia),
           cpp17_input_iterator<const int*>(ia),
           forward_iterator<const int*>(ic),
           forward_iterator<const int*>(ic + 1),
           cuda::std::equal_to<int>())
         == cpp17_input_iterator<const int*>(ia));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
