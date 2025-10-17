/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

// is_unsigned

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_unsigned()
{
  static_assert(cuda::std::is_unsigned<T>::value, "");
  static_assert(cuda::std::is_unsigned<const T>::value, "");
  static_assert(cuda::std::is_unsigned<volatile T>::value, "");
  static_assert(cuda::std::is_unsigned<const volatile T>::value, "");
  static_assert(cuda::std::is_unsigned_v<T>, "");
  static_assert(cuda::std::is_unsigned_v<const T>, "");
  static_assert(cuda::std::is_unsigned_v<volatile T>, "");
  static_assert(cuda::std::is_unsigned_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_unsigned()
{
  static_assert(!cuda::std::is_unsigned<T>::value, "");
  static_assert(!cuda::std::is_unsigned<const T>::value, "");
  static_assert(!cuda::std::is_unsigned<volatile T>::value, "");
  static_assert(!cuda::std::is_unsigned<const volatile T>::value, "");
  static_assert(!cuda::std::is_unsigned_v<T>, "");
  static_assert(!cuda::std::is_unsigned_v<const T>, "");
  static_assert(!cuda::std::is_unsigned_v<volatile T>, "");
  static_assert(!cuda::std::is_unsigned_v<const volatile T>, "");
}

class Class
{
public:
  __host__ __device__ ~Class();
};

struct A; // incomplete

int main(int, char**)
{
  test_is_not_unsigned<void>();
  test_is_not_unsigned<int&>();
  test_is_not_unsigned<Class>();
  test_is_not_unsigned<int*>();
  test_is_not_unsigned<const int*>();
  test_is_not_unsigned<char[3]>();
  test_is_not_unsigned<char[]>();
  test_is_not_unsigned<int>();
  test_is_not_unsigned<double>();
  test_is_not_unsigned<A>();

  test_is_unsigned<bool>();
  test_is_unsigned<unsigned>();

#if _CCCL_HAS_INT128()
  test_is_unsigned<__uint128_t>();
  test_is_not_unsigned<__int128_t>();
#endif // _CCCL_HAS_INT128()

  return 0;
}
