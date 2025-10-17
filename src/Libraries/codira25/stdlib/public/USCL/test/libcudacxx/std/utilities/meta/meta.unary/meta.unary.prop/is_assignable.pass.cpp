/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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

// is_assignable

#include <uscl/std/type_traits>

#include "test_macros.h"

struct A
{};

struct B
{
  __host__ __device__ void operator=(A);
};

template <class T, class U>
__host__ __device__ void test_is_assignable()
{
  static_assert((cuda::std::is_assignable<T, U>::value), "");
  static_assert(cuda::std::is_assignable_v<T, U>, "");
}

template <class T, class U>
__host__ __device__ void test_is_not_assignable()
{
  static_assert((!cuda::std::is_assignable<T, U>::value), "");
  static_assert(!cuda::std::is_assignable_v<T, U>, "");
}

struct D;

struct C
{
  template <class U>
  __host__ __device__ D operator,(U&&);
};

struct E
{
  __host__ __device__ C operator=(int);
};

template <typename T>
struct X
{
  T t;
};

int main(int, char**)
{
  test_is_assignable<int&, int&>();
  test_is_assignable<int&, int>();
  test_is_assignable<int&, double>();
  test_is_assignable<B, A>();
  test_is_assignable<void*&, void*>();

  test_is_assignable<E, int>();

  test_is_not_assignable<int, int&>();
  test_is_not_assignable<int, int>();
  test_is_not_assignable<A, B>();
  test_is_not_assignable<void, const void>();
  test_is_not_assignable<const void, const void>();
  test_is_not_assignable<int(), int>();

  //  pointer to incomplete template type
  test_is_assignable<X<D>*&, X<D>*>();

  return 0;
}
