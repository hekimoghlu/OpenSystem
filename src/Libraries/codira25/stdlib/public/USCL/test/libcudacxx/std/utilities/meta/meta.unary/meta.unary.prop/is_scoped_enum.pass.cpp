/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/type_traits>

// is_scoped_enum

#include <uscl/std/cstddef> // for cuda::std::nullptr_t
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_positive()
{
  static_assert(cuda::std::is_scoped_enum<T>::value, "");
  static_assert(cuda::std::is_scoped_enum<const T>::value, "");
  static_assert(cuda::std::is_scoped_enum<volatile T>::value, "");
  static_assert(cuda::std::is_scoped_enum<const volatile T>::value, "");

  static_assert(cuda::std::is_scoped_enum_v<T>, "");
  static_assert(cuda::std::is_scoped_enum_v<const T>, "");
  static_assert(cuda::std::is_scoped_enum_v<volatile T>, "");
  static_assert(cuda::std::is_scoped_enum_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_negative()
{
  static_assert(!cuda::std::is_scoped_enum<T>::value, "");
  static_assert(!cuda::std::is_scoped_enum<const T>::value, "");
  static_assert(!cuda::std::is_scoped_enum<volatile T>::value, "");
  static_assert(!cuda::std::is_scoped_enum<const volatile T>::value, "");

  static_assert(!cuda::std::is_scoped_enum_v<T>, "");
  static_assert(!cuda::std::is_scoped_enum_v<const T>, "");
  static_assert(!cuda::std::is_scoped_enum_v<volatile T>, "");
  static_assert(!cuda::std::is_scoped_enum_v<const volatile T>, "");
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
enum class CEnum1
{
  zero,
  one
};
enum class CEnum2;
enum class CEnum3 : short;
struct incomplete_type;

using FunctionPtr  = void (*)();
using FunctionType = void();

struct TestMembers
{
  __host__ __device__ static int static_method(int)
  {
    return 0;
  }
  __host__ __device__ int method()
  {
    return 0;
  }

  enum E1
  {
    m_zero,
    m_one
  };
  enum class CE1;
};

__host__ __device__ void func1();
__host__ __device__ int func2(int);

int main(int, char**)
{
  test_positive<CEnum1>();
  test_positive<CEnum2>();
  test_positive<CEnum3>();
  test_positive<TestMembers::CE1>();

  test_negative<Enum>();
  test_negative<TestMembers::E1>();

  test_negative<cuda::std::nullptr_t>();
  test_negative<void>();
  test_negative<int>();
  test_negative<int&>();
  test_negative<int&&>();
  test_negative<int*>();
  test_negative<double>();
  test_negative<const int*>();
  test_negative<char[3]>();
  test_negative<char[]>();
  test_negative<Union>();
  test_negative<Empty>();
  test_negative<bit_zero>();
  test_negative<NotEmpty>();
  test_negative<Abstract>();
  test_negative<FunctionPtr>();
  test_negative<FunctionType>();
  test_negative<incomplete_type>();
  test_negative<int TestMembers::*>();
  test_negative<void (TestMembers::*)()>();

  test_negative<decltype(func1)>();
  test_negative<decltype(&func1)>();
  test_negative<decltype(func2)>();
  test_negative<decltype(&func2)>();
  test_negative<decltype(TestMembers::static_method)>();
  test_negative<decltype(&TestMembers::static_method)>();
  test_negative<decltype(&TestMembers::method)>();

  return 0;
}
