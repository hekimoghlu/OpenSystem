/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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

// Let REQUIRED-SPAN-SIZE(e, strides) be:
//    - 1, if e.rank() == 0 is true,
//    - otherwise 0, if the size of the multidimensional index space e is 0,
//    - otherwise 1 plus the sum of products of (e.extent(r) - 1) and strides[r] for all r in the range [0, e.rank()).

// constexpr index_type required_span_size() const noexcept;
//
//   Returns: REQUIRED-SPAN-SIZE(extents(), strides_).

#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/mdspan>

#include "test_macros.h"

template <class E>
__host__ __device__ constexpr void
test_required_span_size(E e, cuda::std::array<int, E::rank()> strides, typename E::index_type expected_size)
{
  using M = cuda::std::layout_stride::mapping<E>;
  const M m(e, strides);

  static_assert(noexcept(m.required_span_size()));
  assert(m.required_span_size() == expected_size);
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_required_span_size(cuda::std::extents<int>(), cuda::std::array<int, 0>{}, 1);
  test_required_span_size(cuda::std::extents<unsigned, D>(0), cuda::std::array<int, 1>{5}, 0);
  test_required_span_size(cuda::std::extents<unsigned, D>(1), cuda::std::array<int, 1>{5}, 1);
  test_required_span_size(cuda::std::extents<unsigned, D>(7), cuda::std::array<int, 1>{5}, 31);
  test_required_span_size(cuda::std::extents<unsigned, 7>(), cuda::std::array<int, 1>{5}, 31);
  test_required_span_size(cuda::std::extents<unsigned, 7, 8>(), cuda::std::array<int, 2>{20, 2}, 135);
  test_required_span_size(
    cuda::std::extents<int64_t, D, 8, D, D>(7, 9, 10), cuda::std::array<int, 4>{1, 7, 7 * 8, 7 * 8 * 9}, 5040);
  test_required_span_size(
    cuda::std::extents<int64_t, 1, 8, D, D>(9, 10), cuda::std::array<int, 4>{1, 7, 7 * 8, 7 * 8 * 9}, 5034);
  test_required_span_size(
    cuda::std::extents<int64_t, 1, 0, D, D>(9, 10), cuda::std::array<int, 4>{1, 7, 7 * 8, 7 * 8 * 9}, 0);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
