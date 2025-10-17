/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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

// is_same

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_is_same()
{
  static_assert((cuda::std::is_same<T, U>::value), "");
  static_assert((!cuda::std::is_same<const T, U>::value), "");
  static_assert((!cuda::std::is_same<T, const U>::value), "");
  static_assert((cuda::std::is_same<const T, const U>::value), "");
  static_assert((cuda::std::is_same_v<T, U>), "");
  static_assert((!cuda::std::is_same_v<const T, U>), "");
  static_assert((!cuda::std::is_same_v<T, const U>), "");
  static_assert((cuda::std::is_same_v<const T, const U>), "");
}

template <class T, class U>
__host__ __device__ void test_is_same_ref()
{
  static_assert((cuda::std::is_same<T, U>::value), "");
  static_assert((cuda::std::is_same<const T, U>::value), "");
  static_assert((cuda::std::is_same<T, const U>::value), "");
  static_assert((cuda::std::is_same<const T, const U>::value), "");
  static_assert((cuda::std::is_same_v<T, U>), "");
  static_assert((cuda::std::is_same_v<const T, U>), "");
  static_assert((cuda::std::is_same_v<T, const U>), "");
  static_assert((cuda::std::is_same_v<const T, const U>), "");
}

template <class T, class U>
__host__ __device__ void test_is_not_same()
{
  static_assert((!cuda::std::is_same<T, U>::value), "");
}

class Class
{
public:
  __host__ __device__ ~Class();
};

int main(int, char**)
{
  test_is_same<int, int>();
  test_is_same<void, void>();
  test_is_same<Class, Class>();
  test_is_same<int*, int*>();
  test_is_same_ref<int&, int&>();

  test_is_not_same<int, void>();
  test_is_not_same<void, Class>();
  test_is_not_same<Class, int*>();
  test_is_not_same<int*, int&>();
  test_is_not_same<int&, int>();

  return 0;
}
