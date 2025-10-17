/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
//
//===----------------------------------------------------------------------===//
//
// <numeric>

// template <class _Tp>
// _Tp* midpoint(_Tp* __a, _Tp* __b) noexcept
//

#include <uscl/std/cassert>
#include <uscl/std/numeric>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr void constexpr_test()
{
  constexpr T array[1000] = {};
  static_assert(cuda::std::is_same_v<decltype(cuda::std::midpoint(array, array)), const T*>);
  static_assert(noexcept(cuda::std::midpoint(array, array)));

  static_assert(cuda::std::midpoint(array, array) == array, "");
  static_assert(cuda::std::midpoint(array, array + 1000) == array + 500, "");

  static_assert(cuda::std::midpoint(array, array + 9) == array + 4, "");
  static_assert(cuda::std::midpoint(array, array + 10) == array + 5, "");
  static_assert(cuda::std::midpoint(array, array + 11) == array + 5, "");
  static_assert(cuda::std::midpoint(array + 9, array) == array + 5, "");
  static_assert(cuda::std::midpoint(array + 10, array) == array + 5, "");
  static_assert(cuda::std::midpoint(array + 11, array) == array + 6, "");
}

template <typename T>
__host__ __device__ void runtime_test()
{
  T array[1000] = {}; // we need an array to make valid pointers
  static_assert(cuda::std::is_same_v<decltype(cuda::std::midpoint(array, array)), T*>);
  static_assert(noexcept(cuda::std::midpoint(array, array)));

  assert(cuda::std::midpoint(array, array) == array);
  assert(cuda::std::midpoint(array, array + 1000) == array + 500);

  assert(cuda::std::midpoint(array, array + 9) == array + 4);
  assert(cuda::std::midpoint(array, array + 10) == array + 5);
  assert(cuda::std::midpoint(array, array + 11) == array + 5);
  assert(cuda::std::midpoint(array + 9, array) == array + 5);
  assert(cuda::std::midpoint(array + 10, array) == array + 5);
  assert(cuda::std::midpoint(array + 11, array) == array + 6);

  // explicit instantiation
  static_assert(cuda::std::is_same_v<decltype(cuda::std::midpoint<T>(array, array)), T*>);
  static_assert(noexcept(cuda::std::midpoint<T>(array, array)));
  assert(cuda::std::midpoint<T>(array, array) == array);
  assert(cuda::std::midpoint<T>(array, array + 1000) == array + 500);
}

template <typename T>
__host__ __device__ void pointer_test()
{
  runtime_test<T>();
  runtime_test<const T>();
  runtime_test<volatile T>();
  runtime_test<const volatile T>();

  //  The constexpr tests are always const, but we can test them anyway.
  constexpr_test<T>();
  constexpr_test<const T>();

#if !TEST_COMPILER(GCC)
  constexpr_test<volatile T>();
  constexpr_test<const volatile T>();
#endif // !TEST_COMPILER(GCC)
}

int main(int, char**)
{
  pointer_test<char>();
  pointer_test<int>();
  pointer_test<double>();

  return 0;
}
