/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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

// UNSUPPORTED: msvc

// type_traits

// is_base_of

#define _LIBCUDACXX_USE_IS_BASE_OF_FALLBACK

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_is_base_of()
{
  static_assert((cuda::std::is_base_of<T, U>::value), "");
  static_assert((cuda::std::is_base_of<const T, U>::value), "");
  static_assert((cuda::std::is_base_of<T, const U>::value), "");
  static_assert((cuda::std::is_base_of<const T, const U>::value), "");
  static_assert((cuda::std::is_base_of_v<T, U>), "");
  static_assert((cuda::std::is_base_of_v<const T, U>), "");
  static_assert((cuda::std::is_base_of_v<T, const U>), "");
  static_assert((cuda::std::is_base_of_v<const T, const U>), "");
}

template <class T, class U>
__host__ __device__ void test_is_not_base_of()
{
  static_assert((!cuda::std::is_base_of<T, U>::value), "");
}

struct B
{};
struct B1 : B
{};
struct B2 : B
{};
struct D
    : private B1
    , private B2
{};
struct I0; // incomplete

int main(int, char**)
{
  test_is_base_of<B, D>();
  test_is_base_of<B1, D>();
  test_is_base_of<B2, D>();
  test_is_base_of<B, B1>();
  test_is_base_of<B, B2>();
  test_is_base_of<B, B>();

  test_is_not_base_of<D, B>();
  test_is_not_base_of<B&, D&>();
  test_is_not_base_of<B[3], D[3]>();
  test_is_not_base_of<int, int>();

  //  A scalar is never the base class of anything (including incomplete types)
  test_is_not_base_of<int, B>();
  test_is_not_base_of<int, B1>();
  test_is_not_base_of<int, B2>();
  test_is_not_base_of<int, D>();
  test_is_not_base_of<int, I0>();

  //  A scalar never has base classes (including incomplete types)
  test_is_not_base_of<B, int>();
  test_is_not_base_of<B1, int>();
  test_is_not_base_of<B2, int>();
  test_is_not_base_of<D, int>();
  test_is_not_base_of<I0, int>();

  return 0;
}
