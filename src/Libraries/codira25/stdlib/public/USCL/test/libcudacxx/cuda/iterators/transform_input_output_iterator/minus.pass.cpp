/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
// friend constexpr iterator operator-(iterator i, difference_type n);
// friend constexpr difference_type operator-(const iterator& x, const iterator& y);

#include <uscl/iterator>
#include <uscl/std/cassert>
#include <uscl/std/cstdint>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  PlusOne input_func{};
  TimesTwo output_func{};

  { // <iterator> - difference_type
    cuda::transform_input_output_iterator iter1{buffer + 6, input_func, output_func};
    cuda::transform_input_output_iterator iter2{buffer + 6, input_func, output_func};
    assert(iter1 == iter2);
    assert(iter1 - 0 == iter2);
    assert(iter1 - 5 != iter2);
    assert((iter1 - 5).base() == buffer + 1);

    static_assert(noexcept(iter2 - 5));
    static_assert(!cuda::std::is_reference_v<decltype(iter2 - 5)>);
  }

  { // <iterator> - <iterator>
    cuda::transform_input_output_iterator iter1{buffer + 6, input_func, output_func};
    cuda::transform_input_output_iterator iter2{buffer + 3, input_func, output_func};
    assert(iter1 - iter2 == 3);
    assert(iter1 - iter1 == 0);
    assert(iter2 - iter1 == -3);

    static_assert(noexcept(iter1 - iter2));
    static_assert(cuda::std::same_as<decltype(iter1 - iter2), cuda::std::iter_difference_t<int*>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
