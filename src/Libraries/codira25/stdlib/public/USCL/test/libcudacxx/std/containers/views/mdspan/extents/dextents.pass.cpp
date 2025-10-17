/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

// template<class IndexType, size_t Rank>
//     using dextents = see below;
//
// Result: A type E that is a specialization of extents such that
//         E::rank() == Rank && E::rank() == E::rank_dynamic() is true,
//         and E::index_type denotes IndexType.

#include <uscl/std/cstddef>
#include <uscl/std/mdspan>

#include "test_macros.h"

template <class IndexType>
__host__ __device__ void test_alias_template_dextents()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  static_assert(cuda::std::is_same_v<cuda::std::dextents<IndexType, 0>, cuda::std::extents<IndexType>>);
  static_assert(cuda::std::is_same_v<cuda::std::dextents<IndexType, 1>, cuda::std::extents<IndexType, D>>);
  static_assert(cuda::std::is_same_v<cuda::std::dextents<IndexType, 2>, cuda::std::extents<IndexType, D, D>>);
  static_assert(cuda::std::is_same_v<cuda::std::dextents<IndexType, 3>, cuda::std::extents<IndexType, D, D, D>>);
  static_assert(
    cuda::std::is_same_v<cuda::std::dextents<IndexType, 9>, cuda::std::extents<IndexType, D, D, D, D, D, D, D, D, D>>);
}

int main(int, char**)
{
  test_alias_template_dextents<int>();
  test_alias_template_dextents<unsigned int>();
  test_alias_template_dextents<size_t>();
  return 0;
}
