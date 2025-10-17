/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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

// template<class Iter, IntegralLike Size, Callable Generator>
//   requires OutputIterator<Iter, Generator::result_type>
//         && CopyConstructible<Generator>
//   constexpr void      // constexpr after c++17
//   generate_n(Iter first, Size n, Generator gen);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "user_defined_integral.h"

struct gen_test
{
  constexpr __host__ __device__ int operator()() const noexcept
  {
    return 1;
  }
};

template <class Iter, class Size>
constexpr __host__ __device__ void test()
{
  constexpr int N = 5;
  int ia[N + 1]   = {0};
  assert(cuda::std::generate_n(Iter(ia), Size(N), gen_test()) == Iter(ia + N));
  for (int i = 0; i < N; ++i)
  {
    assert(ia[i] == 1);
  }

  for (int i = N; i < N + 1; ++i)
  {
    assert(ia[i] == 0);
  }
}

template <class Iter>
constexpr __host__ __device__ void test()
{
  test<Iter, int>();
  test<Iter, unsigned int>();
  test<Iter, long>();
  test<Iter, unsigned long>();
  test<Iter, UserDefinedIntegral<unsigned>>();
  test<Iter, float>();
  test<Iter, double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<Iter, long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
}

constexpr __host__ __device__ bool test()
{
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
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
