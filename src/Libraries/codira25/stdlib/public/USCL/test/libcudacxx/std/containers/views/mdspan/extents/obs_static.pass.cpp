/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

// static constexpr rank_type rank() noexcept;
// static constexpr rank_type rank_dynamic() noexcept;
//
// static constexpr size_t static_extent(rank_type i) noexcept;
//
//   Preconditions: i < rank() is true.
//
//   Returns: Ei.
//
//
// constexpr index_type extent(rank_type i) const noexcept;
//
//   Preconditions: i < rank() is true.
//
//   Returns: Di.
//

#include <uscl/std/cassert>
#include <uscl/std/mdspan>
#include <uscl/std/utility>

#include "test_macros.h"

template <class E,
          size_t rank,
          size_t rank_dynamic,
          size_t... StaticExts,
          size_t... Indices,
          cuda::std::enable_if_t<(rank > 0), int> = 0>
__host__ __device__ void
test_static_observers(cuda::std::index_sequence<StaticExts...>, cuda::std::index_sequence<Indices...>)
{
  static_assert(noexcept(E::rank()));
  static_assert(E::rank() == rank, "");
  static_assert(noexcept(E::rank_dynamic()));
  static_assert(E::rank_dynamic() == rank_dynamic, "");

  // Let's only test this if the call isn't a precondition violation
  static_assert(noexcept(E::static_extent(0)));
  static_assert(cuda::std::is_same_v<decltype(E::static_extent(0)), size_t>);
  static_assert(cuda::std::__all < E::static_extent(Indices) == StaticExts... > ::value, "");
}

template <class E,
          size_t rank,
          size_t rank_dynamic,
          size_t... StaticExts,
          size_t... Indices,
          cuda::std::enable_if_t<(rank == 0), int> = 0>
__host__ __device__ void
test_static_observers(cuda::std::index_sequence<StaticExts...>, cuda::std::index_sequence<Indices...>)
{
  static_assert(noexcept(E::rank()));
  static_assert(E::rank() == rank, "");
  static_assert(noexcept(E::rank_dynamic()));
  static_assert(E::rank_dynamic() == rank_dynamic, "");
}

template <class E, size_t rank, size_t rank_dynamic, size_t... StaticExts>
__host__ __device__ void test_static_observers()
{
  test_static_observers<E, rank, rank_dynamic>(
    cuda::std::index_sequence<StaticExts...>(), cuda::std::make_index_sequence<sizeof...(StaticExts)>());
}

template <class T>
__host__ __device__ void test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  constexpr size_t S                  = 5;

  test_static_observers<cuda::std::extents<T>, 0, 0>();

  test_static_observers<cuda::std::extents<T, S>, 1, 0, S>();
  test_static_observers<cuda::std::extents<T, D>, 1, 1, D>();

  test_static_observers<cuda::std::extents<T, S, S>, 2, 0, S, S>();
  test_static_observers<cuda::std::extents<T, S, D>, 2, 1, S, D>();
  test_static_observers<cuda::std::extents<T, D, S>, 2, 1, D, S>();
  test_static_observers<cuda::std::extents<T, D, D>, 2, 2, D, D>();

  test_static_observers<cuda::std::extents<T, S, S, S>, 3, 0, S, S, S>();
  test_static_observers<cuda::std::extents<T, S, S, D>, 3, 1, S, S, D>();
  test_static_observers<cuda::std::extents<T, S, D, S>, 3, 1, S, D, S>();
  test_static_observers<cuda::std::extents<T, D, S, S>, 3, 1, D, S, S>();
  test_static_observers<cuda::std::extents<T, S, D, D>, 3, 2, S, D, D>();
  test_static_observers<cuda::std::extents<T, D, S, D>, 3, 2, D, S, D>();
  test_static_observers<cuda::std::extents<T, D, D, S>, 3, 2, D, D, S>();
  test_static_observers<cuda::std::extents<T, D, D, D>, 3, 3, D, D, D>();
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
