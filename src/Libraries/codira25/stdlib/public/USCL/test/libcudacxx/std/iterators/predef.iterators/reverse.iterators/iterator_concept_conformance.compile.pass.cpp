/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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

// reverse_iterator

#include <uscl/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class I1>
__host__ __device__ constexpr bool common_reverse_iterator_checks()
{
  static_assert(cuda::std::indirectly_writable<I1, int>);
  static_assert(cuda::std::sentinel_for<I1, I1>);
  return true;
}

using reverse_bidirectional_iterator = cuda::std::reverse_iterator<bidirectional_iterator<int*>>;
static_assert(common_reverse_iterator_checks<reverse_bidirectional_iterator>());
static_assert(cuda::std::bidirectional_iterator<reverse_bidirectional_iterator>);
static_assert(!cuda::std::random_access_iterator<reverse_bidirectional_iterator>);
static_assert(!cuda::std::sized_sentinel_for<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert(cuda::std::indirectly_movable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert(cuda::std::indirectly_movable_storable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert(cuda::std::indirectly_copyable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert(cuda::std::indirectly_copyable_storable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert(cuda::std::indirectly_swappable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);

using reverse_random_access_iterator = cuda::std::reverse_iterator<random_access_iterator<int*>>;
static_assert(common_reverse_iterator_checks<reverse_random_access_iterator>());
static_assert(cuda::std::random_access_iterator<reverse_random_access_iterator>);
static_assert(!cuda::std::contiguous_iterator<reverse_random_access_iterator>);
static_assert(cuda::std::sized_sentinel_for<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert(cuda::std::indirectly_movable<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert(cuda::std::indirectly_movable_storable<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert(cuda::std::indirectly_copyable<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert(cuda::std::indirectly_copyable_storable<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert(cuda::std::indirectly_swappable<reverse_random_access_iterator, reverse_random_access_iterator>);

using reverse_contiguous_iterator = cuda::std::reverse_iterator<contiguous_iterator<int*>>;
static_assert(common_reverse_iterator_checks<reverse_contiguous_iterator>());
static_assert(cuda::std::random_access_iterator<reverse_contiguous_iterator>);
static_assert(!cuda::std::contiguous_iterator<reverse_contiguous_iterator>);
static_assert(cuda::std::sized_sentinel_for<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert(cuda::std::indirectly_movable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert(cuda::std::indirectly_movable_storable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert(cuda::std::indirectly_copyable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert(cuda::std::indirectly_copyable_storable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert(cuda::std::indirectly_swappable<reverse_contiguous_iterator, reverse_contiguous_iterator>);

static_assert(
  cuda::std::equality_comparable_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<const int*>>);
static_assert(
  !cuda::std::equality_comparable_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<char*>>);
static_assert(
  cuda::std::totally_ordered_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<const int*>>);
static_assert(!cuda::std::totally_ordered_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<char*>>);
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
static_assert(
  cuda::std::three_way_comparable_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<const int*>>);
static_assert(
  !cuda::std::three_way_comparable_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<char*>>);
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

int main(int, char**)
{
  return 0;
}
