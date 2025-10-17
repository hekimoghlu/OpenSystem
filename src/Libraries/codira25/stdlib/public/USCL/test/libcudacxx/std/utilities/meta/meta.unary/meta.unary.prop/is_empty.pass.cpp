/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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

// is_empty

// T is a non-union class type with:
//  no non-static data members,
//  no unnamed bit-fields of non-zero length,
//  no virtual member functions,
//  no virtual base classes,
//  and no base class B for which is_empty_v<B> is false.

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_empty()
{
  static_assert(cuda::std::is_empty<T>::value, "");
  static_assert(cuda::std::is_empty<const T>::value, "");
  static_assert(cuda::std::is_empty<volatile T>::value, "");
  static_assert(cuda::std::is_empty<const volatile T>::value, "");
  static_assert(cuda::std::is_empty_v<T>, "");
  static_assert(cuda::std::is_empty_v<const T>, "");
  static_assert(cuda::std::is_empty_v<volatile T>, "");
  static_assert(cuda::std::is_empty_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_empty()
{
  static_assert(!cuda::std::is_empty<T>::value, "");
  static_assert(!cuda::std::is_empty<const T>::value, "");
  static_assert(!cuda::std::is_empty<volatile T>::value, "");
  static_assert(!cuda::std::is_empty<const volatile T>::value, "");
  static_assert(!cuda::std::is_empty_v<T>, "");
  static_assert(!cuda::std::is_empty_v<const T>, "");
  static_assert(!cuda::std::is_empty_v<volatile T>, "");
  static_assert(!cuda::std::is_empty_v<const volatile T>, "");
}

class Empty
{};
struct NotEmpty
{
  int foo;
};

class VirtualFn
{
  __host__ __device__ virtual ~VirtualFn();
};

union Union
{};

struct EmptyBase : public Empty
{};
struct VirtualBase : virtual Empty
{};
struct NotEmptyBase : public NotEmpty
{};

struct StaticMember
{
  static const int foo;
};
struct NonStaticMember
{
  int foo;
};

struct bit_zero
{
  int : 0;
};

struct bit_one
{
  int : 1;
};

int main(int, char**)
{
  test_is_not_empty<void>();
  test_is_not_empty<int&>();
  test_is_not_empty<int>();
  test_is_not_empty<double>();
  test_is_not_empty<int*>();
  test_is_not_empty<const int*>();
  test_is_not_empty<char[3]>();
  test_is_not_empty<char[]>();
  test_is_not_empty<Union>();
  test_is_not_empty<NotEmpty>();
  test_is_not_empty<VirtualFn>();
  test_is_not_empty<VirtualBase>();
  test_is_not_empty<NotEmptyBase>();
  test_is_not_empty<NonStaticMember>();
  //    test_is_not_empty<bit_one>();

  test_is_empty<Empty>();
  test_is_empty<EmptyBase>();
  test_is_empty<StaticMember>();
  test_is_empty<bit_zero>();

  return 0;
}
