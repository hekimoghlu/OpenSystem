/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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

// is_member_pointer

#include <uscl/std/cstddef> // for cuda::std::nullptr_t
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_member_pointer()
{
  static_assert(cuda::std::is_member_pointer<T>::value, "");
  static_assert(cuda::std::is_member_pointer<const T>::value, "");
  static_assert(cuda::std::is_member_pointer<volatile T>::value, "");
  static_assert(cuda::std::is_member_pointer<const volatile T>::value, "");
  static_assert(cuda::std::is_member_pointer_v<T>, "");
  static_assert(cuda::std::is_member_pointer_v<const T>, "");
  static_assert(cuda::std::is_member_pointer_v<volatile T>, "");
  static_assert(cuda::std::is_member_pointer_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_member_pointer()
{
  static_assert(!cuda::std::is_member_pointer<T>::value, "");
  static_assert(!cuda::std::is_member_pointer<const T>::value, "");
  static_assert(!cuda::std::is_member_pointer<volatile T>::value, "");
  static_assert(!cuda::std::is_member_pointer<const volatile T>::value, "");
  static_assert(!cuda::std::is_member_pointer_v<T>, "");
  static_assert(!cuda::std::is_member_pointer_v<const T>, "");
  static_assert(!cuda::std::is_member_pointer_v<volatile T>, "");
  static_assert(!cuda::std::is_member_pointer_v<const volatile T>, "");
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

  test_is_member_pointer<int Empty::*>();
  test_is_member_pointer<void (Empty::*)(int)>();

  test_is_not_member_pointer<cuda::std::nullptr_t>();
  test_is_not_member_pointer<void>();
  test_is_not_member_pointer<void*>();
  test_is_not_member_pointer<int>();
  test_is_not_member_pointer<int*>();
  test_is_not_member_pointer<const int*>();
  test_is_not_member_pointer<int&>();
  test_is_not_member_pointer<int&&>();
  test_is_not_member_pointer<double>();
  test_is_not_member_pointer<char[3]>();
  test_is_not_member_pointer<char[]>();
  test_is_not_member_pointer<Union>();
  test_is_not_member_pointer<Empty>();
  test_is_not_member_pointer<incomplete_type>();
  test_is_not_member_pointer<bit_zero>();
  test_is_not_member_pointer<NotEmpty>();
  test_is_not_member_pointer<Abstract>();
  test_is_not_member_pointer<int(int)>();
  test_is_not_member_pointer<Enum>();
  test_is_not_member_pointer<FunctionPtr>();

  return 0;
}
