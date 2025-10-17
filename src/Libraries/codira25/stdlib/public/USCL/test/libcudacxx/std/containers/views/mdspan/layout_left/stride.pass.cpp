/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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

// constexpr index_type stride(rank_type i) const noexcept;
//
//   Constraints: extents_type::rank() > 0 is true.
//
//   Preconditions: i < extents_type::rank() is true.
//
//   Returns: extents().rev-prod-of-extents(i).

#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/mdspan>

#include "test_macros.h"

template <class E, class... Args>
__host__ __device__ constexpr void test_stride(cuda::std::array<typename E::index_type, E::rank()> strides, Args... args)
{
  cuda::std::layout_left::mapping<E> m{E{args...}};

  static_assert(noexcept(m.stride(0)));
  for (size_t r = 0; r < E::rank(); r++)
  {
    assert(strides[r] == m.stride(r));
  }
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_stride<cuda::std::extents<unsigned, D>>(cuda::std::array<unsigned, 1>{1}, 7);
  test_stride<cuda::std::extents<unsigned, 7>>(cuda::std::array<unsigned, 1>{1});
  test_stride<cuda::std::extents<unsigned, 7, 8>>(cuda::std::array<unsigned, 2>{1, 7});
  test_stride<cuda::std::extents<int64_t, D, 8, D, D>>(cuda::std::array<int64_t, 4>{1, 7, 56, 504}, 7, 9, 10);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
