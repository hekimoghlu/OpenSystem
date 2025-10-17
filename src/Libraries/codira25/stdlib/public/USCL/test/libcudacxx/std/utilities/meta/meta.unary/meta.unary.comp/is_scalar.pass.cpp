/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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

// is_scalar

#include <uscl/std/cstddef> // for cuda::std::nullptr_t
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_scalar()
{
  static_assert(cuda::std::is_scalar<T>::value, "");
  static_assert(cuda::std::is_scalar<const T>::value, "");
  static_assert(cuda::std::is_scalar<volatile T>::value, "");
  static_assert(cuda::std::is_scalar<const volatile T>::value, "");
  static_assert(cuda::std::is_scalar_v<T>, "");
  static_assert(cuda::std::is_scalar_v<const T>, "");
  static_assert(cuda::std::is_scalar_v<volatile T>, "");
  static_assert(cuda::std::is_scalar_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_scalar()
{
  static_assert(!cuda::std::is_scalar<T>::value, "");
  static_assert(!cuda::std::is_scalar<const T>::value, "");
  static_assert(!cuda::std::is_scalar<volatile T>::value, "");
  static_assert(!cuda::std::is_scalar<const volatile T>::value, "");
  static_assert(!cuda::std::is_scalar_v<T>, "");
  static_assert(!cuda::std::is_scalar_v<const T>, "");
  static_assert(!cuda::std::is_scalar_v<volatile T>, "");
  static_assert(!cuda::std::is_scalar_v<const volatile T>, "");
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
  //  Arithmetic types (3.9.1), enumeration types, pointer types, pointer to member types (3.9.2),
  //    cuda::std::nullptr_t, and cv-qualified versions of these types (3.9.3)
  //    are collectively called scalar types.

  test_is_scalar<cuda::std::nullptr_t>();
  test_is_scalar<short>();
  test_is_scalar<unsigned short>();
  test_is_scalar<int>();
  test_is_scalar<unsigned int>();
  test_is_scalar<long>();
  test_is_scalar<unsigned long>();
  test_is_scalar<bool>();
  test_is_scalar<char>();
  test_is_scalar<signed char>();
  test_is_scalar<unsigned char>();
  test_is_scalar<wchar_t>();
  test_is_scalar<double>();
  test_is_scalar<int*>();
  test_is_scalar<const int*>();
  test_is_scalar<int Empty::*>();
  test_is_scalar<void (Empty::*)(int)>();
  test_is_scalar<Enum>();
  test_is_scalar<FunctionPtr>();

  test_is_not_scalar<void>();
  test_is_not_scalar<int&>();
  test_is_not_scalar<int&&>();
  test_is_not_scalar<char[3]>();
  test_is_not_scalar<char[]>();
  test_is_not_scalar<Union>();
  test_is_not_scalar<Empty>();
  test_is_not_scalar<incomplete_type>();
  test_is_not_scalar<bit_zero>();
  test_is_not_scalar<NotEmpty>();
  test_is_not_scalar<Abstract>();
  test_is_not_scalar<int(int)>();

  return 0;
}
