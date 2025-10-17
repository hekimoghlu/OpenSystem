/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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

// type_traits

// has_nothrow_move_constructor

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_nothrow_move_constructible()
{
  static_assert(cuda::std::is_nothrow_move_constructible<T>::value, "");
  static_assert(cuda::std::is_nothrow_move_constructible<const T>::value, "");
  static_assert(cuda::std::is_nothrow_move_constructible_v<T>, "");
  static_assert(cuda::std::is_nothrow_move_constructible_v<const T>, "");
}

template <class T>
__host__ __device__ void test_has_not_nothrow_move_constructor()
{
#if !TEST_COMPILER(NVHPC)
  static_assert(!cuda::std::is_nothrow_move_constructible<T>::value, "");
  static_assert(!cuda::std::is_nothrow_move_constructible<const T>::value, "");
#endif // !TEST_COMPILER(NVHPC)
  static_assert(!cuda::std::is_nothrow_move_constructible<volatile T>::value, "");
  static_assert(!cuda::std::is_nothrow_move_constructible<const volatile T>::value, "");
#if !TEST_COMPILER(NVHPC)
  static_assert(!cuda::std::is_nothrow_move_constructible_v<T>, "");
  static_assert(!cuda::std::is_nothrow_move_constructible_v<const T>, "");
#endif // TEST_COMPILER(NVHPC)
  static_assert(!cuda::std::is_nothrow_move_constructible_v<volatile T>, "");
  static_assert(!cuda::std::is_nothrow_move_constructible_v<const volatile T>, "");
}

class Empty
{};

union Union
{};

struct bit_zero
{
  int : 0;
};

struct A
{
  __host__ __device__ A(const A&);
};

int main(int, char**)
{
  test_has_not_nothrow_move_constructor<void>();
  test_has_not_nothrow_move_constructor<A>();

  test_is_nothrow_move_constructible<int&>();
  test_is_nothrow_move_constructible<Union>();
  test_is_nothrow_move_constructible<Empty>();
  test_is_nothrow_move_constructible<int>();
  test_is_nothrow_move_constructible<double>();
  test_is_nothrow_move_constructible<int*>();
  test_is_nothrow_move_constructible<const int*>();
  test_is_nothrow_move_constructible<bit_zero>();

  return 0;
}
