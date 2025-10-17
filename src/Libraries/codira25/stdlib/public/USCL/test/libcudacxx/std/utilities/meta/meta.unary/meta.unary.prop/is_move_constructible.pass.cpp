/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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

// is_move_constructible

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_move_constructible()
{
  static_assert(cuda::std::is_move_constructible<T>::value, "");
  static_assert(cuda::std::is_move_constructible_v<T>, "");
}

template <class T>
__host__ __device__ void test_is_not_move_constructible()
{
  static_assert(!cuda::std::is_move_constructible<T>::value, "");
  static_assert(!cuda::std::is_move_constructible_v<T>, "");
}

class Empty
{};

class NotEmpty
{
public:
  __host__ __device__ virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

class Abstract
{
public:
  __host__ __device__ virtual ~Abstract() = 0;
};

struct A
{
  __host__ __device__ A(const A&);
};

struct B
{
  __host__ __device__ B(B&&);
};

int main(int, char**)
{
#if !TEST_COMPILER(GCC) || TEST_STD_VER < 2020
  test_is_not_move_constructible<char[3]>();
  test_is_not_move_constructible<char[]>();
#endif // !TEST_COMPILER(GCC) || TEST_STD_VER < 2020
  test_is_not_move_constructible<void>();
  test_is_not_move_constructible<Abstract>();

  test_is_move_constructible<A>();
  test_is_move_constructible<int&>();
  test_is_move_constructible<Union>();
  test_is_move_constructible<Empty>();
  test_is_move_constructible<int>();
  test_is_move_constructible<double>();
  test_is_move_constructible<int*>();
  test_is_move_constructible<const int*>();
  test_is_move_constructible<NotEmpty>();
  test_is_move_constructible<bit_zero>();
  test_is_move_constructible<B>();

  return 0;
}
