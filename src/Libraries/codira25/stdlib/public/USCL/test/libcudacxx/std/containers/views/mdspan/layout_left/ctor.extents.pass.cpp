/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// constexpr mapping(const extents_type&) noexcept;
//
// Preconditions: The size of the multidimensional index space e is representable
//                as a value of type index_type ([basic.fundamental]).
//
// Effects: Direct-non-list-initializes extents_ with e.

#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/mdspan>

#include "test_macros.h"

template <class E, cuda::std::enable_if_t<E::rank() != 0, int> = 0>
__host__ __device__ constexpr void test_construction(E e)
{
  using M = cuda::std::layout_left::mapping<E>;
  static_assert(noexcept(M{e}));
  M m(e);

  // check correct extents are returned
  static_assert(noexcept(m.extents()));
  assert(m.extents() == e);

  // check required_span_size()
  typename E::index_type expected_size = 1;
  for (typename E::rank_type r = 0; r < E::rank(); r++)
  {
    expected_size *= e.extent(r);
  }
  assert(m.required_span_size() == expected_size);
}

template <class E, cuda::std::enable_if_t<E::rank() == 0, int> = 0>
__host__ __device__ constexpr void test_construction(E e)
{
  using M = cuda::std::layout_left::mapping<E>;
  static_assert(noexcept(M{e}));
  M m(e);

  // check correct extents are returned
  static_assert(noexcept(m.extents()));
  assert(m.extents() == e);
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_construction(cuda::std::extents<int>());
  test_construction(cuda::std::extents<unsigned, D>(7));
  test_construction(cuda::std::extents<unsigned, 7>());
  test_construction(cuda::std::extents<unsigned, 7, 8>());
  test_construction(cuda::std::extents<int64_t, D, 8, D, D>(7, 9, 10));
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
