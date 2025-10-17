/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 6, 2022.
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

//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

// <cuda/std/span>

// template <class It, class End>
// constexpr explicit(Extent != dynamic_extent) span(It first, End last);
// Requires: [first, last) shall be a valid range.
//   If Extent is not equal to dynamic_extent, then last - first shall be equal to Extent.
//

#include <uscl/std/cassert>
#include <uscl/std/span>

#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Sentinel>
__host__ __device__ constexpr bool test_ctor()
{
  T val[2] = {};
  auto s1  = cuda::std::span<T>(cuda::std::begin(val), Sentinel(cuda::std::end(val)));
  auto s2  = cuda::std::span<T, 2>(cuda::std::begin(val), Sentinel(cuda::std::end(val)));
  assert(s1.data() == cuda::std::data(val) && s1.size() == cuda::std::size(val));
  assert(s2.data() == cuda::std::data(val) && s2.size() == cuda::std::size(val));
  return true;
}

template <size_t Extent>
__host__ __device__ constexpr void test_constructibility()
{
  static_assert(cuda::std::is_constructible_v<cuda::std::span<int, Extent>, int*, int*>, "");
  static_assert(!cuda::std::is_constructible_v<cuda::std::span<int, Extent>, const int*, const int*>, "");
  static_assert(!cuda::std::is_constructible_v<cuda::std::span<int, Extent>, volatile int*, volatile int*>, "");
  static_assert(cuda::std::is_constructible_v<cuda::std::span<const int, Extent>, int*, int*>, "");
  static_assert(cuda::std::is_constructible_v<cuda::std::span<const int, Extent>, const int*, const int*>, "");
  static_assert(!cuda::std::is_constructible_v<cuda::std::span<const int, Extent>, volatile int*, volatile int*>, "");
  static_assert(cuda::std::is_constructible_v<cuda::std::span<volatile int, Extent>, int*, int*>, "");
  static_assert(!cuda::std::is_constructible_v<cuda::std::span<volatile int, Extent>, const int*, const int*>, "");
  static_assert(cuda::std::is_constructible_v<cuda::std::span<volatile int, Extent>, volatile int*, volatile int*>, "");
  static_assert(!cuda::std::is_constructible_v<cuda::std::span<int, Extent>, int*, float*>, ""); // types wrong
}

__host__ __device__ constexpr bool test()
{
  test_constructibility<cuda::std::dynamic_extent>();
  test_constructibility<3>();
  struct A
  {};
  assert((test_ctor<int, int*>()));
  // assert((test_ctor<int, sized_sentinel<int*>>()));
  assert((test_ctor<A, A*>()));
  // assert((test_ctor<A, sized_sentinel<A*>>()));
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
