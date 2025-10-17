/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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

// template<class IndexType, size_t... Extents>
// class extents {
// public:
//  // types
//  using index_type = IndexType;
//  using size_type = make_unsigned_t<index_type>;
//  using rank_type = size_t;
//
//  static constexpr rank_type rank() noexcept { return sizeof...(Extents); }
//  static constexpr rank_type rank_dynamic() noexcept { return dynamic-index(rank()); }
//  ...
//  }

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/mdspan>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <size_t... Extents>
__host__ __device__ constexpr size_t count_dynamic_extents()
{
  constexpr size_t arr[] = {Extents...};
  size_t res             = 0;
  for (size_t i = 0; i < sizeof...(Extents); ++i)
  {
    res += static_cast<size_t>(arr[i] == cuda::std::dynamic_extent);
  }
  return res;
}

template <class E, class IndexType, size_t... Extents>
__host__ __device__ void testExtents()
{
  static_assert(cuda::std::is_same_v<typename E::index_type, IndexType>);
  static_assert(cuda::std::is_same_v<typename E::size_type, cuda::std::make_unsigned_t<IndexType>>);
  static_assert(cuda::std::is_same_v<typename E::rank_type, size_t>);

  static_assert(sizeof...(Extents) == E::rank(), "");
  static_assert(count_dynamic_extents<Extents...>() == E::rank_dynamic());

  static_assert(cuda::std::regular<E>, "");
  static_assert(cuda::std::is_trivially_copyable<E>::value, "");

  static_assert(cuda::std::is_empty<E>::value == (E::rank_dynamic() == 0), "");
}

template <class IndexType, size_t... Extents>
__host__ __device__ void testExtents()
{
  testExtents<cuda::std::extents<IndexType, Extents...>, IndexType, Extents...>();
}

template <class T>
__host__ __device__ void test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  testExtents<T, D>();
  testExtents<T, 3>();
  testExtents<T, 3, 3>();
  testExtents<T, 3, D>();
  testExtents<T, D, 3>();
  testExtents<T, D, D>();
  testExtents<T, 3, 3, 3>();
  testExtents<T, 3, 3, D>();
  testExtents<T, 3, D, D>();
  testExtents<T, D, 3, D>();
  testExtents<T, D, D, D>();
  testExtents<T, 3, D, 3>();
  testExtents<T, D, 3, 3>();
  testExtents<T, D, D, 3>();

  testExtents<T, 9, 8, 7, 6, 5, 4, 3, 2, 1>();
  testExtents<T, 9, D, 7, 6, D, D, 3, D, D>();
  testExtents<T, D, D, D, D, D, D, D, D, D>();
}

int main(int, char**)
{
  test<int>();
  test<unsigned>();
  test<char>();
  test<long long>();
  test<size_t>();
  return 0;
}
