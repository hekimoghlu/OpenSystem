/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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

// template<class In, class Out>
// concept indirectly_movable;

#include <uscl/std/iterator>

#include "MoveOnly.h"
#include "test_macros.h"

// Can move between pointers.
static_assert(cuda::std::indirectly_movable<int*, int*>, "");
static_assert(cuda::std::indirectly_movable<const int*, int*>, "");
static_assert(!cuda::std::indirectly_movable<int*, const int*>, "");
static_assert(cuda::std::indirectly_movable<const int*, int*>, "");

// Can move from a pointer into an array but arrays aren't considered indirectly movable-from.
#if !TEST_COMPILER(MSVC) || TEST_STD_VER != 2017
static_assert(cuda::std::indirectly_movable<int*, int[2]>, "");
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER != 2017
static_assert(!cuda::std::indirectly_movable<int[2], int*>, "");
static_assert(!cuda::std::indirectly_movable<int[2], int[2]>, "");
static_assert(!cuda::std::indirectly_movable<int (&)[2], int (&)[2]>, "");

// Can't move between non-pointer types.
static_assert(!cuda::std::indirectly_movable<int*, int>, "");
static_assert(!cuda::std::indirectly_movable<int, int*>, "");
static_assert(!cuda::std::indirectly_movable<int, int>, "");

// Check some less common types.
static_assert(!cuda::std::indirectly_movable<void*, void*>, "");
static_assert(!cuda::std::indirectly_movable<int*, void*>, "");
static_assert(!cuda::std::indirectly_movable<int(), int()>, "");
static_assert(!cuda::std::indirectly_movable<int*, int()>, "");
static_assert(!cuda::std::indirectly_movable<void, void>, "");

// Can move move-only objects.
static_assert(cuda::std::indirectly_movable<MoveOnly*, MoveOnly*>, "");
static_assert(!cuda::std::indirectly_movable<MoveOnly*, const MoveOnly*>, "");
static_assert(!cuda::std::indirectly_movable<const MoveOnly*, const MoveOnly*>, "");
static_assert(!cuda::std::indirectly_movable<const MoveOnly*, MoveOnly*>, "");

template <class T>
struct PointerTo
{
  using value_type = T;
  __host__ __device__ T& operator*() const;
};

// Can copy through a dereferenceable class.
static_assert(cuda::std::indirectly_movable<int*, PointerTo<int>>, "");
static_assert(!cuda::std::indirectly_movable<int*, PointerTo<const int>>, "");
static_assert(cuda::std::indirectly_copyable<PointerTo<int>, PointerTo<int>>, "");
static_assert(!cuda::std::indirectly_copyable<PointerTo<int>, PointerTo<const int>>, "");
static_assert(cuda::std::indirectly_movable<MoveOnly*, PointerTo<MoveOnly>>, "");
static_assert(cuda::std::indirectly_movable<PointerTo<MoveOnly>, MoveOnly*>, "");
static_assert(cuda::std::indirectly_movable<PointerTo<MoveOnly>, PointerTo<MoveOnly>>, "");

int main(int, char**)
{
  return 0;
}
