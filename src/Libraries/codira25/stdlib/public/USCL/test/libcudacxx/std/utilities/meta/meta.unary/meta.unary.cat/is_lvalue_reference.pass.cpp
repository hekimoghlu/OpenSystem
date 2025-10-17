/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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

// is_lvalue_reference

#include <uscl/std/cstddef> // for cuda::std::nullptr_t
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_lvalue_reference()
{
  static_assert(cuda::std::is_lvalue_reference<T>::value, "");
  static_assert(cuda::std::is_lvalue_reference<const T>::value, "");
  static_assert(cuda::std::is_lvalue_reference<volatile T>::value, "");
  static_assert(cuda::std::is_lvalue_reference<const volatile T>::value, "");
  static_assert(cuda::std::is_lvalue_reference_v<T>, "");
  static_assert(cuda::std::is_lvalue_reference_v<const T>, "");
  static_assert(cuda::std::is_lvalue_reference_v<volatile T>, "");
  static_assert(cuda::std::is_lvalue_reference_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_lvalue_reference()
{
  static_assert(!cuda::std::is_lvalue_reference<T>::value, "");
  static_assert(!cuda::std::is_lvalue_reference<const T>::value, "");
  static_assert(!cuda::std::is_lvalue_reference<volatile T>::value, "");
  static_assert(!cuda::std::is_lvalue_reference<const volatile T>::value, "");
  static_assert(!cuda::std::is_lvalue_reference_v<T>, "");
  static_assert(!cuda::std::is_lvalue_reference_v<const T>, "");
  static_assert(!cuda::std::is_lvalue_reference_v<volatile T>, "");
  static_assert(!cuda::std::is_lvalue_reference_v<const volatile T>, "");
}

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
struct incomplete_type;

typedef void (*FunctionPtr)();

int main(int, char**)
{
  test_is_lvalue_reference<int&>();

  test_is_not_lvalue_reference<cuda::std::nullptr_t>();
  test_is_not_lvalue_reference<void>();
  test_is_not_lvalue_reference<int>();
  test_is_not_lvalue_reference<int*>();
  test_is_not_lvalue_reference<int&&>();
  test_is_not_lvalue_reference<double>();
  test_is_not_lvalue_reference<const int*>();
  test_is_not_lvalue_reference<char[3]>();
  test_is_not_lvalue_reference<char[]>();
  test_is_not_lvalue_reference<Union>();
  test_is_not_lvalue_reference<Enum>();
  test_is_not_lvalue_reference<FunctionPtr>();
  test_is_not_lvalue_reference<Empty>();
  test_is_not_lvalue_reference<bit_zero>();
  test_is_not_lvalue_reference<NotEmpty>();
  test_is_not_lvalue_reference<Abstract>();
  test_is_not_lvalue_reference<incomplete_type>();

  return 0;
}
