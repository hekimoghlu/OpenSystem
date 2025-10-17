/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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

// template<BidirectionalIterator Iter, Predicate<auto, Iter::value_type> Pred>
//   requires ShuffleIterator<Iter>
//         && CopyConstructible<Pred>
//   constexpr Iter  // constexpr in C++20
//   partition(Iter first, Iter last, Pred pred);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct is_odd
{
  __host__ __device__ constexpr bool operator()(const int& i) const
  {
    return i & 1;
  }
};

template <class Iter>
__host__ __device__ constexpr void test()
{
  // check mixed
  int ia[]          = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  Iter r            = cuda::std::partition(Iter(ia), Iter(ia + sa), is_odd());
  assert(base(r) == ia + 5);
  for (int* i = ia; i < base(r); ++i)
  {
    assert(is_odd()(*i));
  }
  for (int* i = base(r); i < ia + sa; ++i)
  {
    assert(!is_odd()(*i));
  }
  // check empty
  r = cuda::std::partition(Iter(ia), Iter(ia), is_odd());
  assert(base(r) == ia);
  // check all false
  for (unsigned i = 0; i < sa; ++i)
  {
    ia[i] = 2 * i;
  }
  r = cuda::std::partition(Iter(ia), Iter(ia + sa), is_odd());
  assert(base(r) == ia);
  // check all true
  for (unsigned i = 0; i < sa; ++i)
  {
    ia[i] = 2 * i + 1;
  }
  r = cuda::std::partition(Iter(ia), Iter(ia + sa), is_odd());
  assert(base(r) == ia + sa);
  // check all true but last
  for (unsigned i = 0; i < sa; ++i)
  {
    ia[i] = 2 * i + 1;
  }
  ia[sa - 1] = 10;
  r          = cuda::std::partition(Iter(ia), Iter(ia + sa), is_odd());
  assert(base(r) == ia + sa - 1);
  for (int* i = ia; i < base(r); ++i)
  {
    assert(is_odd()(*i));
  }
  for (int* i = base(r); i < ia + sa; ++i)
  {
    assert(!is_odd()(*i));
  }
  // check all true but first
  for (unsigned i = 0; i < sa; ++i)
  {
    ia[i] = 2 * i + 1;
  }
  ia[0] = 10;
  r     = cuda::std::partition(Iter(ia), Iter(ia + sa), is_odd());
  assert(base(r) == ia + sa - 1);
  for (int* i = ia; i < base(r); ++i)
  {
    assert(is_odd()(*i));
  }
  for (int* i = base(r); i < ia + sa; ++i)
  {
    assert(!is_odd()(*i));
  }
  // check all false but last
  for (unsigned i = 0; i < sa; ++i)
  {
    ia[i] = 2 * i;
  }
  ia[sa - 1] = 11;
  r          = cuda::std::partition(Iter(ia), Iter(ia + sa), is_odd());
  assert(base(r) == ia + 1);
  for (int* i = ia; i < base(r); ++i)
  {
    assert(is_odd()(*i));
  }
  for (int* i = base(r); i < ia + sa; ++i)
  {
    assert(!is_odd()(*i));
  }
  // check all false but first
  for (unsigned i = 0; i < sa; ++i)
  {
    ia[i] = 2 * i;
  }
  ia[0] = 11;
  r     = cuda::std::partition(Iter(ia), Iter(ia + sa), is_odd());
  assert(base(r) == ia + 1);
  for (int* i = ia; i < base(r); ++i)
  {
    assert(is_odd()(*i));
  }
  for (int* i = base(r); i < ia + sa; ++i)
  {
    assert(!is_odd()(*i));
  }
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
