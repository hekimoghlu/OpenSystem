/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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

// constexpr index_type required_span_size() const noexcept;
//
// Returns: extents().fwd-prod-of-extents(extents_type::rank()).

#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/mdspan>

#include "test_macros.h"

template <class E>
__host__ __device__ constexpr void test_required_span_size(E e, typename E::index_type expected_size)
{
  using M = cuda::std::layout_left::mapping<E>;
  const M m(e);

  static_assert(noexcept(m.required_span_size()));
  assert(m.required_span_size() == expected_size);
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_required_span_size(cuda::std::extents<int>(), 1);
  test_required_span_size(cuda::std::extents<unsigned, D>(0), 0);
  test_required_span_size(cuda::std::extents<unsigned, D>(1), 1);
  test_required_span_size(cuda::std::extents<unsigned, D>(7), 7);
  test_required_span_size(cuda::std::extents<unsigned, 7>(), 7);
  test_required_span_size(cuda::std::extents<unsigned, 7, 8>(), 56);
  test_required_span_size(cuda::std::extents<int64_t, D, 8, D, D>(7, 9, 10), 5040);
  test_required_span_size(cuda::std::extents<int64_t, 1, 8, D, D>(9, 10), 720);
  test_required_span_size(cuda::std::extents<int64_t, 1, 0, D, D>(9, 10), 0);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
