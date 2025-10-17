/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
// constexpr permutation_iterator& operator++();
// constexpr permutation_iterator operator++(int);

#include <uscl/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using baseIter             = random_access_iterator<int*>;
  using indexIter            = random_access_iterator<const int*>;
  using permutation_iterator = cuda::permutation_iterator<baseIter, indexIter>;
  int buffer[]               = {1, 2, 3, 4, 5, 6, 7, 8};
  const int offset[]         = {4, 2, 6};

  const baseIter base{buffer};
  const indexIter off{offset};

  permutation_iterator iter(base, off);
  assert(*iter == buffer[offset[0]]);
  assert(iter++ == permutation_iterator(base, off));
  assert(*iter == buffer[offset[1]]);
  assert(++iter == permutation_iterator(base, off + 2));
  assert(*iter == buffer[offset[2]]);
  assert(iter.index() == offset[2]);

  static_assert(cuda::std::is_same_v<decltype(iter++), permutation_iterator>);
  static_assert(cuda::std::is_same_v<decltype(++iter), permutation_iterator&>);

  // The test iterators are not noexcept
  static_assert(!noexcept(iter++));
  static_assert(!noexcept(++iter));

  // Pointers are noexcept incrementable
  static_assert(noexcept(cuda::permutation_iterator<int*, int*>()++));
  static_assert(noexcept(++cuda::permutation_iterator<int*, int*>()));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
