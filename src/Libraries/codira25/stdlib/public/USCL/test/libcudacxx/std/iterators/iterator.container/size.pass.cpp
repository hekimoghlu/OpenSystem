/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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

// <cuda/std/iterator>
// template <class C> constexpr auto size(const C& c) -> decltype(c.size());         // C++17
// template <class T, size_t N> constexpr size_t size(const T (&array)[N]) noexcept; // C++17

#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/inplace_vector>
#include <uscl/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#  include <cuda/std/list>
#endif
#include <uscl/std/initializer_list>
#include <uscl/std/string_view>

#include "test_macros.h"

template <typename C>
__host__ __device__ void test_container(C& c)
{
  //  Can't say noexcept here because the container might not be
  assert(cuda::std::size(c) == c.size());
}

template <typename C>
__host__ __device__ void test_const_container(const C& c)
{
  //  Can't say noexcept here because the container might not be
  assert(cuda::std::size(c) == c.size());
}

template <typename T>
__host__ __device__ void test_const_container(const cuda::std::initializer_list<T>& c)
{
  static_assert(noexcept(cuda::std::size(c))); // our cuda::std::size is conditionally noexcept
  assert(cuda::std::size(c) == c.size());
}

template <typename T>
__host__ __device__ void test_container(cuda::std::initializer_list<T>& c)
{
  static_assert(noexcept(cuda::std::size(c))); // our cuda::std::size is conditionally noexcept
  assert(cuda::std::size(c) == c.size());
}

template <typename T, size_t Sz>
__host__ __device__ void test_const_array(const T (&array)[Sz])
{
  static_assert(noexcept(cuda::std::size(array)));
  assert(cuda::std::size(array) == Sz);
}

TEST_GLOBAL_VARIABLE constexpr int arrA[]{1, 2, 3};

int main(int, char**)
{
  cuda::std::inplace_vector<int, 3> v;
  v.push_back(1);
#if defined(_LIBCUDACXX_HAS_LIST)
  cuda::std::list<int> l;
  l.push_back(2);
#endif
  cuda::std::array<int, 1> a;
  a[0]                                = 3;
  cuda::std::initializer_list<int> il = {4};
  test_container(v);
#if defined(_LIBCUDACXX_HAS_LIST)
  test_container(l);
#endif
  test_container(a);
  test_container(il);

  test_const_container(v);
#if defined(_LIBCUDACXX_HAS_LIST)
  test_const_container(l);
#endif
  test_const_container(a);
  test_const_container(il);

  cuda::std::string_view sv{"ABC"};
  test_container(sv);
  test_const_container(sv);

  test_const_array(arrA);

  return 0;
}
