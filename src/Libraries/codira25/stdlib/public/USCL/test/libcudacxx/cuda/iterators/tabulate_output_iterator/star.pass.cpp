/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
// constexpr auto operator*() const noexcept(is_nothrow_copy_constructible_v<W>);

#include <uscl/iterator>
#include <uscl/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class Fn>
__host__ __device__ constexpr void test()
{
  Fn func{};

  {
    cuda::tabulate_output_iterator iter{func, 10};
    for (int i = 10; i < 100; ++i, ++iter)
    {
      *iter = i;
    }
    static_assert(noexcept(*iter));
    static_assert(!cuda::std::is_same_v<decltype(*iter), void>);
  }

  {
    const cuda::tabulate_output_iterator iter{func, 10};
    *iter = 10;
    static_assert(noexcept(*iter = 10));
    static_assert(!cuda::std::is_same_v<decltype(*iter), void>);
  }
}

__host__ __device__ constexpr bool test()
{
  test<basic_functor>();
  test<mutable_functor>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
