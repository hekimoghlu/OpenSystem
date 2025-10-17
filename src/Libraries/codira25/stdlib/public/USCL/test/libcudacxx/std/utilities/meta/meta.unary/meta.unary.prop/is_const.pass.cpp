/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

// is_const

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_const()
{
  static_assert(!cuda::std::is_const<T>::value, "");
  static_assert(cuda::std::is_const<const T>::value, "");
  static_assert(!cuda::std::is_const<volatile T>::value, "");
  static_assert(cuda::std::is_const<const volatile T>::value, "");
  static_assert(!cuda::std::is_const_v<T>, "");
  static_assert(cuda::std::is_const_v<const T>, "");
  static_assert(!cuda::std::is_const_v<volatile T>, "");
  static_assert(cuda::std::is_const_v<const volatile T>, "");
}

struct A; // incomplete

int main(int, char**)
{
  test_is_const<void>();
  test_is_const<int>();
  test_is_const<double>();
  test_is_const<int*>();
  test_is_const<const int*>();
  test_is_const<char[3]>();
  test_is_const<char[]>();

  test_is_const<A>();

  static_assert(!cuda::std::is_const<int&>::value, "");
  static_assert(!cuda::std::is_const<const int&>::value, "");

  return 0;
}
