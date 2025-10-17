/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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

// keep this test in sync with `is_floating_point.pass.cpp` for `cuda::is_floating_point`

// type_traits

// is_floating_point

#include <uscl/std/cstddef> // for cuda::std::nullptr_t
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_floating_point()
{
  static_assert(cuda::std::is_floating_point<T>::value, "");
  static_assert(cuda::std::is_floating_point<const T>::value, "");
  static_assert(cuda::std::is_floating_point<volatile T>::value, "");
  static_assert(cuda::std::is_floating_point<const volatile T>::value, "");
  static_assert(cuda::std::is_floating_point_v<T>, "");
  static_assert(cuda::std::is_floating_point_v<const T>, "");
  static_assert(cuda::std::is_floating_point_v<volatile T>, "");
  static_assert(cuda::std::is_floating_point_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_floating_point()
{
  static_assert(!cuda::std::is_floating_point<T>::value, "");
  static_assert(!cuda::std::is_floating_point<const T>::value, "");
  static_assert(!cuda::std::is_floating_point<volatile T>::value, "");
  static_assert(!cuda::std::is_floating_point<const volatile T>::value, "");
  static_assert(!cuda::std::is_floating_point_v<T>, "");
  static_assert(!cuda::std::is_floating_point_v<const T>, "");
  static_assert(!cuda::std::is_floating_point_v<volatile T>, "");
  static_assert(!cuda::std::is_floating_point_v<const volatile T>, "");
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
  test_is_floating_point<float>();
  test_is_floating_point<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_is_floating_point<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  test_is_not_floating_point<short>();
  test_is_not_floating_point<unsigned short>();
  test_is_not_floating_point<int>();
  test_is_not_floating_point<unsigned int>();
  test_is_not_floating_point<long>();
  test_is_not_floating_point<unsigned long>();

  test_is_not_floating_point<cuda::std::nullptr_t>();
  test_is_not_floating_point<void>();
  test_is_not_floating_point<int&>();
  test_is_not_floating_point<int&&>();
  test_is_not_floating_point<int*>();
  test_is_not_floating_point<const int*>();
  test_is_not_floating_point<char[3]>();
  test_is_not_floating_point<char[]>();
  test_is_not_floating_point<Union>();
  test_is_not_floating_point<Empty>();
  test_is_not_floating_point<bit_zero>();
  test_is_not_floating_point<NotEmpty>();
  test_is_not_floating_point<Abstract>();
  test_is_not_floating_point<Enum>();
  test_is_not_floating_point<FunctionPtr>();
  test_is_not_floating_point<incomplete_type>();

  return 0;
}
