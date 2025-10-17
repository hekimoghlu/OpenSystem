/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
// <algorithm>

// template<RandomAccessIterator Iter>
//   requires ShuffleIterator<Iter> && LessThanComparable<Iter::value_type>
//   constexpr void  // constexpr in C++20
//   make_heap(Iter first, Iter last, Compare comp);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>
#include <uscl/std/functional>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Iter>
__host__ __device__ constexpr void test()
{
  int orig[15] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  T work[15]   = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  for (int n = 0; n < 15; ++n)
  {
    cuda::std::make_heap(Iter(work), Iter(work + n), cuda::std::greater<T>());
    assert(cuda::std::is_heap(work, work + n, cuda::std::greater<T>()));
    assert(cuda::std::is_permutation(work, work + n, orig));
    cuda::std::copy(orig, orig + n, work);
  }

  {
    T input[] = {3, 4, 1, 2, 5};
    cuda::std::make_heap(Iter(input), Iter(input + 5), cuda::std::greater<T>());
    assert(cuda::std::is_heap(input, input + 5, cuda::std::greater<T>()));
    cuda::std::pop_heap(input, input + 5, cuda::std::greater<T>());
    assert(input[4] == 1);
    cuda::std::pop_heap(input, input + 4, cuda::std::greater<T>());
    assert(input[3] == 2);
    cuda::std::pop_heap(input, input + 3, cuda::std::greater<T>());
    assert(input[2] == 3);
    cuda::std::pop_heap(input, input + 2, cuda::std::greater<T>());
    assert(input[1] == 4);
    cuda::std::pop_heap(input, input + 1, cuda::std::greater<T>());
    assert(input[0] == 5);
  }
}

__host__ __device__ constexpr bool test()
{
  test<int, random_access_iterator<int*>>();
  test<int, int*>();
  test<MoveOnly, random_access_iterator<MoveOnly*>>();
  test<MoveOnly, MoveOnly*>();

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  return 0;
}
