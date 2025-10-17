/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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

// is_compound

#include <uscl/std/cstddef> // for cuda::std::nullptr_t
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_compound()
{
  static_assert(cuda::std::is_compound<T>::value, "");
  static_assert(cuda::std::is_compound<const T>::value, "");
  static_assert(cuda::std::is_compound<volatile T>::value, "");
  static_assert(cuda::std::is_compound<const volatile T>::value, "");
  static_assert(cuda::std::is_compound_v<T>, "");
  static_assert(cuda::std::is_compound_v<const T>, "");
  static_assert(cuda::std::is_compound_v<volatile T>, "");
  static_assert(cuda::std::is_compound_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_compound()
{
  static_assert(!cuda::std::is_compound<T>::value, "");
  static_assert(!cuda::std::is_compound<const T>::value, "");
  static_assert(!cuda::std::is_compound<volatile T>::value, "");
  static_assert(!cuda::std::is_compound<const volatile T>::value, "");
  static_assert(!cuda::std::is_compound_v<T>, "");
  static_assert(!cuda::std::is_compound_v<const T>, "");
  static_assert(!cuda::std::is_compound_v<volatile T>, "");
  static_assert(!cuda::std::is_compound_v<const volatile T>, "");
}

class incomplete_type;

class Empty
{};

class NotEmpty
{
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
  __host__ __device__ virtual ~Abstract() = 0;
};

enum Enum
{
  zero,
  one
};

typedef void (*FunctionPtr)();

int main(int, char**)
{
  test_is_compound<char[3]>();
  test_is_compound<char[]>();
  test_is_compound<void*>();
  test_is_compound<FunctionPtr>();
  test_is_compound<int&>();
  test_is_compound<int&&>();
  test_is_compound<Union>();
  test_is_compound<Empty>();
  test_is_compound<incomplete_type>();
  test_is_compound<bit_zero>();
  test_is_compound<int*>();
  test_is_compound<const int*>();
  test_is_compound<Enum>();
  test_is_compound<NotEmpty>();
  test_is_compound<Abstract>();

  test_is_not_compound<cuda::std::nullptr_t>();
  test_is_not_compound<void>();
  test_is_not_compound<int>();
  test_is_not_compound<double>();

  return 0;
}
