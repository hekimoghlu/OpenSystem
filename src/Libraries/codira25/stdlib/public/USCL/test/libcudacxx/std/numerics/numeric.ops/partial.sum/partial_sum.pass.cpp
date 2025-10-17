/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 7, 2022.
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
// template <InputIterator InIter, OutputIterator<auto, const InIter::value_type&> OutIter>
//   requires HasPlus<InIter::value_type, InIter::reference>
//         && HasAssign<InIter::value_type,
//                      HasPlus<InIter::value_type, InIter::reference>::result_type>
//         && Constructible<InIter::value_type, InIter::reference>
//   OutIter
//   partial_sum(InIter first, InIter last, OutIter result);

#include <uscl/std/cassert>
#include <uscl/std/numeric>

#include "test_iterators.h"
#include "test_macros.h"

template <class InIter, class OutIter>
__host__ __device__ constexpr void test()
{
  int ia[]         = {1, 2, 3, 4, 5};
  int ir[]         = {1, 3, 6, 10, 15};
  const unsigned s = sizeof(ia) / sizeof(ia[0]);
  int ib[s]        = {0};
  OutIter r        = cuda::std::partial_sum(InIter(ia), InIter(ia + s), OutIter(ib));
  assert(base(r) == ib + s);
  for (unsigned i = 0; i < s; ++i)
  {
    assert(ib[i] == ir[i]);
  }
}

__host__ __device__ constexpr bool test()
{
  test<cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, int*>();

  test<forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, int*>();

  test<bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, int*>();

  test<random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, int*>();

  test<const int*, cpp17_output_iterator<int*>>();
  test<const int*, forward_iterator<int*>>();
  test<const int*, bidirectional_iterator<int*>>();
  test<const int*, random_access_iterator<int*>>();
  test<const int*, int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
