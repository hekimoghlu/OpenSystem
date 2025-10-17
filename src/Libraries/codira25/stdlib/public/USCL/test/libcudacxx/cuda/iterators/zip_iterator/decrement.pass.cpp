/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator--() requires all-bidirectional<Const, Views...>;
// constexpr iterator operator--(int) requires all-bidirectional<Const, Views...>;

#include <uscl/iterator>
#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter>
_CCCL_CONCEPT canDecrement = _CCCL_REQUIRES_EXPR((Iter), Iter iter)((--iter), (iter--));

__host__ __device__ constexpr bool test()
{
  int a[]    = {1, 2, 3, 4};
  double b[] = {4.1, 3.2, 4.3};

  { // all random_access_iterator
    cuda::zip_iterator iter{a + 3, random_access_iterator{b + 3}, cuda::counting_iterator{3}};
    using Iter = decltype(iter);

    static_assert(cuda::std::is_same_v<decltype(--iter), Iter&>);
    auto& it_ref = --iter;
    assert(cuda::std::addressof(it_ref) == cuda::std::addressof(iter));

    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[2]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[2]));
    assert(cuda::std::get<2>(*iter) == 2);

    static_assert(cuda::std::is_same_v<decltype(iter--), Iter>);
    iter--;
    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[1]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[1]));
    assert(cuda::std::get<2>(*iter) == 1);
  }

  { // all bidirectional_iterator
    cuda::zip_iterator iter{a + 3, bidirectional_iterator{b + 3}, cuda::counting_iterator{3}};
    using Iter = decltype(iter);

    static_assert(cuda::std::is_same_v<decltype(--iter), Iter&>);
    auto& it_ref = --iter;
    assert(cuda::std::addressof(it_ref) == cuda::std::addressof(iter));

    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[2]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[2]));
    assert(cuda::std::get<2>(*iter) == 2);

    static_assert(cuda::std::is_same_v<decltype(iter--), Iter>);
    iter--;
    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[1]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[1]));
    assert(cuda::std::get<2>(*iter) == 1);
  }

  { // not all bidirectional_iterator
    cuda::zip_iterator iter{a + 3, cpp17_input_iterator{b + 3}, cuda::counting_iterator{3}};
    using Iter = decltype(iter);

    static_assert(!canDecrement<Iter>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
