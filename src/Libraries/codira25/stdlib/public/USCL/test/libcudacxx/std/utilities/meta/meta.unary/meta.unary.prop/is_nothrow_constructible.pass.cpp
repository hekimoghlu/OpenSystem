/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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
//   struct is_nothrow_constructible;

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_nothrow_constructible()
{
  static_assert((cuda::std::is_nothrow_constructible<T>::value), "");
  static_assert((cuda::std::is_nothrow_constructible_v<T>), "");
}

template <class T, class A0>
__host__ __device__ void test_is_nothrow_constructible()
{
  static_assert((cuda::std::is_nothrow_constructible<T, A0>::value), "");
  static_assert((cuda::std::is_nothrow_constructible_v<T, A0>), "");
}

template <class T>
__host__ __device__ void test_is_not_nothrow_constructible()
{
  static_assert((!cuda::std::is_nothrow_constructible<T>::value), "");
  static_assert((!cuda::std::is_nothrow_constructible_v<T>), "");
}

template <class T, class A0>
__host__ __device__ void test_is_not_nothrow_constructible()
{
  static_assert((!cuda::std::is_nothrow_constructible<T, A0>::value), "");
  static_assert((!cuda::std::is_nothrow_constructible_v<T, A0>), "");
}

template <class T, class A0, class A1>
__host__ __device__ void test_is_not_nothrow_constructible()
{
  static_assert((!cuda::std::is_nothrow_constructible<T, A0, A1>::value), "");
  static_assert((!cuda::std::is_nothrow_constructible_v<T, A0, A1>), "");
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

struct A
{
  __host__ __device__ A(const A&);
};

struct C
{
  __host__ __device__ C(C&); // not const
  __host__ __device__ void operator=(C&); // not const
};

struct Tuple
{
  __host__ __device__ Tuple(Empty&&) noexcept {}
};

int main(int, char**)
{
  test_is_nothrow_constructible<int>();
  test_is_nothrow_constructible<int, const int&>();
  test_is_nothrow_constructible<Empty>();
  test_is_nothrow_constructible<Empty, const Empty&>();

  test_is_not_nothrow_constructible<A, int>();
  test_is_not_nothrow_constructible<A, int, double>();
  test_is_not_nothrow_constructible<A>();
  test_is_not_nothrow_constructible<C>();
  test_is_nothrow_constructible<Tuple&&, Empty>(); // See bug #19616.

  static_assert(!cuda::std::is_constructible<Tuple&, Empty>::value, "");
  test_is_not_nothrow_constructible<Tuple&, Empty>(); // See bug #19616.

  return 0;
}
