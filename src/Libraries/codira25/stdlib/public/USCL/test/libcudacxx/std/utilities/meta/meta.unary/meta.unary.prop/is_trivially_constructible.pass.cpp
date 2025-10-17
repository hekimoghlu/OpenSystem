/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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

// template <class T, class... Args>
//   struct is_trivially_constructible;

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_trivially_constructible()
{
  static_assert((cuda::std::is_trivially_constructible<T>::value), "");
  static_assert((cuda::std::is_trivially_constructible_v<T>), "");
}

template <class T, class A0>
__host__ __device__ void test_is_trivially_constructible()
{
  static_assert((cuda::std::is_trivially_constructible<T, A0>::value), "");
  static_assert((cuda::std::is_trivially_constructible_v<T, A0>), "");
}

template <class T>
__host__ __device__ void test_is_not_trivially_constructible()
{
  static_assert((!cuda::std::is_trivially_constructible<T>::value), "");
  static_assert((!cuda::std::is_trivially_constructible_v<T>), "");
}

template <class T, class A0>
__host__ __device__ void test_is_not_trivially_constructible()
{
  static_assert((!cuda::std::is_trivially_constructible<T, A0>::value), "");
  static_assert((!cuda::std::is_trivially_constructible_v<T, A0>), "");
}

template <class T, class A0, class A1>
__host__ __device__ void test_is_not_trivially_constructible()
{
  static_assert((!cuda::std::is_trivially_constructible<T, A0, A1>::value), "");
  static_assert((!cuda::std::is_trivially_constructible_v<T, A0, A1>), "");
}

struct A
{
  __host__ __device__ explicit A(int);
  __host__ __device__ A(int, double);
};

int main(int, char**)
{
  test_is_trivially_constructible<int>();
  test_is_trivially_constructible<int, const int&>();

  test_is_not_trivially_constructible<A, int>();
  test_is_not_trivially_constructible<A, int, double>();
  test_is_not_trivially_constructible<A>();

  return 0;
}
