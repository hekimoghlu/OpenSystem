/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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

// void_t

// XFAIL: gcc-5.1, gcc-5.2

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test1()
{
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<volatile T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const volatile T>>);
}

template <class T, class U>
__host__ __device__ void test2()
{
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<T, U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const T, U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<volatile T, U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const volatile T, U>>);

  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<U, T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<U, const T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<U, volatile T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<U, const volatile T>>);

  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<T, const U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const T, const U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<volatile T, const U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const volatile T, const U>>);
}

class Class
{
public:
  __host__ __device__ ~Class();
};

int main(int, char**)
{
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<>>);

  test1<void>();
  test1<int>();
  test1<double>();
  test1<int&>();
  test1<Class>();
  test1<Class[]>();
  test1<Class[5]>();

  test2<void, int>();
  test2<double, int>();
  test2<int&, int>();
  test2<Class&, bool>();
  test2<void*, int&>();

  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<int, double const&, Class, volatile int[], void>>);

  return 0;
}
