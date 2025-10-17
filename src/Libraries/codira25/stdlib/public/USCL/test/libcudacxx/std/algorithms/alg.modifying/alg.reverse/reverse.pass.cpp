/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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

// template<BidirectionalIterator Iter>
//   requires HasSwap<Iter::reference, Iter::reference>
//   constexpr void  // constexpr in C++20
//   reverse(Iter first, Iter last);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
__host__ __device__ constexpr void test()
{
  int ia[]          = {0};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  cuda::std::reverse(Iter(ia), Iter(ia));
  assert(ia[0] == 0);
  cuda::std::reverse(Iter(ia), Iter(ia + sa));
  assert(ia[0] == 0);

  int ib[]          = {0, 1};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  cuda::std::reverse(Iter(ib), Iter(ib + sb));
  assert(ib[0] == 1);
  assert(ib[1] == 0);

  int ic[]          = {0, 1, 2};
  const unsigned sc = sizeof(ic) / sizeof(ic[0]);
  cuda::std::reverse(Iter(ic), Iter(ic + sc));
  assert(ic[0] == 2);
  assert(ic[1] == 1);
  assert(ic[2] == 0);

  int id[]          = {0, 1, 2, 3};
  const unsigned sd = sizeof(id) / sizeof(id[0]);
  cuda::std::reverse(Iter(id), Iter(id + sd));
  assert(id[0] == 3);
  assert(id[1] == 2);
  assert(id[2] == 1);
  assert(id[3] == 0);
}

__host__ __device__ constexpr bool test()
{
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
