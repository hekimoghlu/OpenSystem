/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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

// rank

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T, unsigned A>
__host__ __device__ void test_rank()
{
  static_assert(cuda::std::rank<T>::value == A, "");
  static_assert(cuda::std::rank<const T>::value == A, "");
  static_assert(cuda::std::rank<volatile T>::value == A, "");
  static_assert(cuda::std::rank<const volatile T>::value == A, "");
  static_assert(cuda::std::rank_v<T> == A, "");
  static_assert(cuda::std::rank_v<const T> == A, "");
  static_assert(cuda::std::rank_v<volatile T> == A, "");
  static_assert(cuda::std::rank_v<const volatile T> == A, "");
}

class Class
{
public:
  __host__ __device__ ~Class();
};

int main(int, char**)
{
  test_rank<void, 0>();
  test_rank<int&, 0>();
  test_rank<Class, 0>();
  test_rank<int*, 0>();
  test_rank<const int*, 0>();
  test_rank<int, 0>();
  test_rank<double, 0>();
  test_rank<bool, 0>();
  test_rank<unsigned, 0>();

  test_rank<char[3], 1>();
  test_rank<char[][3], 2>();
  test_rank<char[][4][3], 3>();

  return 0;
}
