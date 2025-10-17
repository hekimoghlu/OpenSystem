/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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

// <span>

// template <class It>
// constexpr explicit(Extent != dynamic_extent) span(It first, size_type count);
//  If Extent is not equal to dynamic_extent, then count shall be equal to Extent.
//

#include <uscl/std/cassert>
#include <uscl/std/iterator>
#include <uscl/std/span>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <size_t Extent>
__host__ __device__ constexpr bool test_constructibility()
{
  struct Other
  {};
  static_assert(cuda::std::is_constructible<cuda::std::span<int, Extent>, int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, const int*, size_t>::value, "");
  static_assert(cuda::std::is_constructible<cuda::std::span<const int, Extent>, int*, size_t>::value, "");
  static_assert(cuda::std::is_constructible<cuda::std::span<const int, Extent>, const int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, volatile int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, const volatile int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<const int, Extent>, volatile int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<const int, Extent>, const volatile int*, size_t>::value,
                "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<volatile int, Extent>, const int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<volatile int, Extent>, const volatile int*, size_t>::value,
                "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, double*, size_t>::value, ""); // iterator
                                                                                                         // type differs
                                                                                                         // from span
                                                                                                         // type
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, size_t, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, Other*, size_t>::value, ""); // unrelated
                                                                                                        // iterator type

  return true;
}

template <class T>
__host__ __device__ constexpr bool test_ctor()
{
  T val[2] = {};
  auto s1  = cuda::std::span<T>(val, 2);
  auto s2  = cuda::std::span<T, 2>(val, 2);
  assert(s1.data() == val && s1.size() == 2);
  assert(s2.data() == val && s2.size() == 2);
  return true;
}

__host__ __device__ constexpr bool test()
{
  test_constructibility<cuda::std::dynamic_extent>();
  test_constructibility<3>();

  struct A
  {};
  test_ctor<int>();
  test_ctor<A>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
