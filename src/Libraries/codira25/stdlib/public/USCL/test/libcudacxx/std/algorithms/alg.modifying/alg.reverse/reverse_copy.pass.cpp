/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<BidirectionalIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   constexpr OutIter          // constexpr after C++17
//   reverse_copy(InIter first, InIter last, OutIter result);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

template <class InIter, class OutIter>
__host__ __device__ constexpr void test()
{
  const int ia[]    = {0};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  int ja[sa]        = {-1};
  OutIter r         = cuda::std::reverse_copy(InIter(ia), InIter(ia), OutIter(ja));
  assert(base(r) == ja);
  assert(ja[0] == -1);
  r = cuda::std::reverse_copy(InIter(ia), InIter(ia + sa), OutIter(ja));
  assert(ja[0] == 0);

  const int ib[]    = {0, 1};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  int jb[sb]        = {-1};
  r                 = cuda::std::reverse_copy(InIter(ib), InIter(ib + sb), OutIter(jb));
  assert(base(r) == jb + sb);
  assert(jb[0] == 1);
  assert(jb[1] == 0);

  const int ic[]    = {0, 1, 2};
  const unsigned sc = sizeof(ic) / sizeof(ic[0]);
  int jc[sc]        = {-1};
  r                 = cuda::std::reverse_copy(InIter(ic), InIter(ic + sc), OutIter(jc));
  assert(base(r) == jc + sc);
  assert(jc[0] == 2);
  assert(jc[1] == 1);
  assert(jc[2] == 0);

  int id[]          = {0, 1, 2, 3};
  const unsigned sd = sizeof(id) / sizeof(id[0]);
  int jd[sd]        = {-1};
  r                 = cuda::std::reverse_copy(InIter(id), InIter(id + sd), OutIter(jd));
  assert(base(r) == jd + sd);
  assert(jd[0] == 3);
  assert(jd[1] == 2);
  assert(jd[2] == 1);
  assert(jd[3] == 0);
}

__host__ __device__ constexpr bool test()
{
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
