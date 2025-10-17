/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// move_iterator

#include <uscl/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ void test()
{
  {
    using iterator = cuda::std::move_iterator<cpp17_input_iterator<int*>>;

    static_assert(!cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::input_iterator<iterator>);
    static_assert(!cuda::std::forward_iterator<iterator>);
    static_assert(!cuda::std::sentinel_for<iterator, iterator>); // not copyable
    static_assert(!cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = cuda::std::move_iterator<cpp20_input_iterator<int*>>;

    static_assert(!cuda::std::default_initializable<iterator>);
    static_assert(!cuda::std::copyable<iterator>);
    static_assert(cuda::std::input_iterator<iterator>);
    static_assert(!cuda::std::forward_iterator<iterator>);
    static_assert(!cuda::std::sentinel_for<iterator, iterator>); // not copyable
    static_assert(!cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = cuda::std::move_iterator<forward_iterator<int*>>;

    static_assert(cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::forward_iterator<iterator>);
    static_assert(!cuda::std::bidirectional_iterator<iterator>);
    static_assert(cuda::std::sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = cuda::std::move_iterator<bidirectional_iterator<int*>>;

    static_assert(cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::bidirectional_iterator<iterator>);
    static_assert(!cuda::std::random_access_iterator<iterator>);
    static_assert(cuda::std::sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = cuda::std::move_iterator<random_access_iterator<int*>>;

    static_assert(cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::random_access_iterator<iterator>);
    static_assert(!cuda::std::contiguous_iterator<iterator>);
    static_assert(cuda::std::sentinel_for<iterator, iterator>);
    static_assert(cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = cuda::std::move_iterator<contiguous_iterator<int*>>;

    static_assert(cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::random_access_iterator<iterator>);
    static_assert(!cuda::std::contiguous_iterator<iterator>);
    static_assert(cuda::std::sentinel_for<iterator, iterator>);
    static_assert(cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = cuda::std::move_iterator<int*>;

    static_assert(cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::random_access_iterator<iterator>);
    static_assert(!cuda::std::contiguous_iterator<iterator>);
    static_assert(cuda::std::sentinel_for<iterator, iterator>);
    static_assert(cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
  }
}

int main(int, char**)
{
  return 0;
}
