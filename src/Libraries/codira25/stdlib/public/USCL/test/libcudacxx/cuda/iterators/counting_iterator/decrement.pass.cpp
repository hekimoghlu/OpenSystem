/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
// constexpr iterator& operator--() requires decrementable<W>;
// constexpr iterator operator--(int) requires decrementable<W>;

#include <uscl/iterator>
#include <uscl/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class T>
_CCCL_CONCEPT Decrementable = _CCCL_REQUIRES_EXPR((T), T i)((--i), (i--));

__host__ __device__ constexpr bool test()
{
  {
    cuda::counting_iterator<int> iter1{0};
    cuda::counting_iterator<int> iter2{0};
    assert(iter1 == iter2);
    assert(--iter1 != iter2--);
    assert(iter1 == iter2);

    static_assert(noexcept(--iter2));
    static_assert(noexcept(iter2--));
    static_assert(!cuda::std::is_reference_v<decltype(iter2--)>);
    static_assert(cuda::std::is_reference_v<decltype(--iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(--iter2)>, decltype(iter2--)>);
  }

  { // With a decrementable type
    cuda::counting_iterator<SomeInt> iter1{SomeInt{0}};
    cuda::counting_iterator<SomeInt> iter2{SomeInt{0}};
    assert(iter1 == iter2);
    assert(--iter1 != iter2--);
    assert(iter1 == iter2);

    static_assert(!noexcept(--iter2));
    static_assert(!noexcept(iter2--));
    static_assert(!cuda::std::is_reference_v<decltype(iter2--)>);
    static_assert(cuda::std::is_reference_v<decltype(--iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(--iter2)>, decltype(iter2--)>);
  }

  static_assert(!Decrementable<cuda::counting_iterator<Int42<ValueCtor>>>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
