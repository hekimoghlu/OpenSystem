/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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

// UNSUPPORTED: gcc-6

// <cuda/std/tuple>

// template <class T> constexpr size_t tuple_size_v = tuple_size<T>::value;

#include <uscl/std/tuple>
#include <uscl/std/utility>
// cuda::std::array not supported
// #include <uscl/std/array>

#include "test_macros.h"

template <class Tuple, int Expect>
__host__ __device__ void test()
{
  static_assert(cuda::std::tuple_size_v<Tuple> == Expect, "");
  static_assert(cuda::std::tuple_size_v<Tuple> == cuda::std::tuple_size<Tuple>::value, "");
  static_assert(cuda::std::tuple_size_v<Tuple const> == cuda::std::tuple_size<Tuple>::value, "");
  static_assert(cuda::std::tuple_size_v<Tuple volatile> == cuda::std::tuple_size<Tuple>::value, "");
  static_assert(cuda::std::tuple_size_v<Tuple const volatile> == cuda::std::tuple_size<Tuple>::value, "");
}

int main(int, char**)
{
  test<cuda::std::tuple<>, 0>();

  test<cuda::std::tuple<int>, 1>();
  // cuda::std::array not supported
  // test<cuda::std::array<int, 1>, 1>();

  test<cuda::std::tuple<int, int>, 2>();
  test<cuda::std::pair<int, int>, 2>();
  // cuda::std::array not supported
  // test<cuda::std::array<int, 2>, 2>();

  test<cuda::std::tuple<int, int, int>, 3>();
  // cuda::std::array not supported
  // test<cuda::std::array<int, 3>, 3>();

  return 0;
}
