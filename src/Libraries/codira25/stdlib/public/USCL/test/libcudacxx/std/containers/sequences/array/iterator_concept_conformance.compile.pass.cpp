/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// iterator, const_iterator, reverse_iterator, const_reverse_iterator

#include <uscl/std/array>
#include <uscl/std/iterator>

using iterator               = cuda::std::array<int, 10>::iterator;
using const_iterator         = cuda::std::array<int, 10>::const_iterator;
using reverse_iterator       = cuda::std::array<int, 10>::reverse_iterator;
using const_reverse_iterator = cuda::std::array<int, 10>::const_reverse_iterator;

static_assert(cuda::std::contiguous_iterator<iterator>);
static_assert(cuda::std::indirectly_writable<iterator, int>);
static_assert(cuda::std::sentinel_for<iterator, iterator>);
static_assert(cuda::std::sentinel_for<iterator, const_iterator>);
static_assert(!cuda::std::sentinel_for<iterator, reverse_iterator>);
static_assert(!cuda::std::sentinel_for<iterator, const_reverse_iterator>);
static_assert(cuda::std::sized_sentinel_for<iterator, iterator>);
static_assert(cuda::std::sized_sentinel_for<iterator, const_iterator>);
static_assert(!cuda::std::sized_sentinel_for<iterator, reverse_iterator>);
static_assert(!cuda::std::sized_sentinel_for<iterator, const_reverse_iterator>);
static_assert(cuda::std::indirectly_movable<iterator, iterator>);
static_assert(cuda::std::indirectly_movable_storable<iterator, iterator>);
static_assert(!cuda::std::indirectly_movable<iterator, const_iterator>);
static_assert(!cuda::std::indirectly_movable_storable<iterator, const_iterator>);
static_assert(cuda::std::indirectly_movable<iterator, reverse_iterator>);
static_assert(cuda::std::indirectly_movable_storable<iterator, reverse_iterator>);
static_assert(!cuda::std::indirectly_movable<iterator, const_reverse_iterator>);
static_assert(!cuda::std::indirectly_movable_storable<iterator, const_reverse_iterator>);
static_assert(cuda::std::indirectly_copyable<iterator, iterator>);
static_assert(cuda::std::indirectly_copyable_storable<iterator, iterator>);
static_assert(!cuda::std::indirectly_copyable<iterator, const_iterator>);
static_assert(!cuda::std::indirectly_copyable_storable<iterator, const_iterator>);
static_assert(cuda::std::indirectly_copyable<iterator, reverse_iterator>);
static_assert(cuda::std::indirectly_copyable_storable<iterator, reverse_iterator>);
static_assert(!cuda::std::indirectly_copyable<iterator, const_reverse_iterator>);
static_assert(!cuda::std::indirectly_copyable_storable<iterator, const_reverse_iterator>);
static_assert(cuda::std::indirectly_swappable<iterator, iterator>);

static_assert(cuda::std::contiguous_iterator<const_iterator>);
static_assert(!cuda::std::indirectly_writable<const_iterator, int>);
static_assert(cuda::std::sentinel_for<const_iterator, iterator>);
static_assert(cuda::std::sentinel_for<const_iterator, const_iterator>);
static_assert(!cuda::std::sentinel_for<const_iterator, reverse_iterator>);
static_assert(!cuda::std::sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(cuda::std::sized_sentinel_for<const_iterator, iterator>);
static_assert(cuda::std::sized_sentinel_for<const_iterator, const_iterator>);
static_assert(!cuda::std::sized_sentinel_for<const_iterator, reverse_iterator>);
static_assert(!cuda::std::sized_sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(cuda::std::indirectly_movable<const_iterator, iterator>);
static_assert(cuda::std::indirectly_movable_storable<const_iterator, iterator>);
static_assert(!cuda::std::indirectly_movable<const_iterator, const_iterator>);
static_assert(!cuda::std::indirectly_movable_storable<const_iterator, const_iterator>);
static_assert(cuda::std::indirectly_movable<const_iterator, reverse_iterator>);
static_assert(cuda::std::indirectly_movable_storable<const_iterator, reverse_iterator>);
static_assert(!cuda::std::indirectly_movable<const_iterator, const_reverse_iterator>);
static_assert(!cuda::std::indirectly_movable_storable<const_iterator, const_reverse_iterator>);
static_assert(cuda::std::indirectly_copyable<const_iterator, iterator>);
static_assert(cuda::std::indirectly_copyable_storable<const_iterator, iterator>);
static_assert(!cuda::std::indirectly_copyable<const_iterator, const_iterator>);
static_assert(!cuda::std::indirectly_copyable_storable<const_iterator, const_iterator>);
static_assert(cuda::std::indirectly_copyable<const_iterator, reverse_iterator>);
static_assert(cuda::std::indirectly_copyable_storable<const_iterator, reverse_iterator>);
static_assert(!cuda::std::indirectly_copyable<const_iterator, const_reverse_iterator>);
static_assert(!cuda::std::indirectly_copyable_storable<const_iterator, const_reverse_iterator>);
static_assert(!cuda::std::indirectly_swappable<const_iterator, const_iterator>);

int main(int, char**)
{
  return 0;
}
