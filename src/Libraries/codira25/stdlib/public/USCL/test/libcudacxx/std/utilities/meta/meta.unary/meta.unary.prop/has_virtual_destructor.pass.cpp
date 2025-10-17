/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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

// has_virtual_destructor

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_has_virtual_destructor()
{
  static_assert(cuda::std::has_virtual_destructor<T>::value, "");
  static_assert(cuda::std::has_virtual_destructor<const T>::value, "");
  static_assert(cuda::std::has_virtual_destructor<volatile T>::value, "");
  static_assert(cuda::std::has_virtual_destructor<const volatile T>::value, "");
  static_assert(cuda::std::has_virtual_destructor_v<T>, "");
  static_assert(cuda::std::has_virtual_destructor_v<const T>, "");
  static_assert(cuda::std::has_virtual_destructor_v<volatile T>, "");
  static_assert(cuda::std::has_virtual_destructor_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_has_not_virtual_destructor()
{
  static_assert(!cuda::std::has_virtual_destructor<T>::value, "");
  static_assert(!cuda::std::has_virtual_destructor<const T>::value, "");
  static_assert(!cuda::std::has_virtual_destructor<volatile T>::value, "");
  static_assert(!cuda::std::has_virtual_destructor<const volatile T>::value, "");
  static_assert(!cuda::std::has_virtual_destructor_v<T>, "");
  static_assert(!cuda::std::has_virtual_destructor_v<const T>, "");
  static_assert(!cuda::std::has_virtual_destructor_v<volatile T>, "");
  static_assert(!cuda::std::has_virtual_destructor_v<const volatile T>, "");
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

struct A
{
  __host__ __device__ ~A();
};

int main(int, char**)
{
  test_has_not_virtual_destructor<void>();
  test_has_not_virtual_destructor<A>();
  test_has_not_virtual_destructor<int&>();
  test_has_not_virtual_destructor<Union>();
  test_has_not_virtual_destructor<Empty>();
  test_has_not_virtual_destructor<int>();
  test_has_not_virtual_destructor<double>();
  test_has_not_virtual_destructor<int*>();
  test_has_not_virtual_destructor<const int*>();
  test_has_not_virtual_destructor<char[3]>();
  test_has_not_virtual_destructor<char[]>();
  test_has_not_virtual_destructor<bit_zero>();

  test_has_virtual_destructor<Abstract>();
  test_has_virtual_destructor<NotEmpty>();

  return 0;
}
