/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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
//

// <cuda/std/type_traits>

#include <uscl/std/type_traits>

#include "test_macros.h"

struct A
{};
struct B
{
public:
  __host__ __device__ operator A()
  {
    return a;
  }
  A a;
};

class C
{};
class D
{
public:
  __host__ __device__ operator C() noexcept
  {
    return c;
  }
  C c;
};

int main(int, char**)
{
  static_assert((cuda::std::is_nothrow_convertible<int, double>::value), "");
  static_assert(!(cuda::std::is_nothrow_convertible<int, char*>::value), "");

  static_assert(!(cuda::std::is_nothrow_convertible<A, B>::value), "");
  static_assert((cuda::std::is_nothrow_convertible<D, C>::value), "");

  static_assert((cuda::std::is_nothrow_convertible_v<int, double>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int, char*>), "");

  static_assert(!(cuda::std::is_nothrow_convertible_v<A, B>), "");
  static_assert((cuda::std::is_nothrow_convertible_v<D, C>), "");

  static_assert((cuda::std::is_nothrow_convertible_v<const void, void>), "");
  static_assert((cuda::std::is_nothrow_convertible_v<volatile void, void>), "");
  static_assert((cuda::std::is_nothrow_convertible_v<void, const void>), "");
  static_assert((cuda::std::is_nothrow_convertible_v<void, volatile void>), "");

  static_assert(!(cuda::std::is_nothrow_convertible_v<int[], double[]>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int[], int[]>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int[10], int[10]>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int[10], double[10]>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int[5], double[10]>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int[10], A[10]>), "");

  typedef void V();
  typedef int I();
  static_assert(!(cuda::std::is_nothrow_convertible_v<V, V>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<V, I>), "");

  return 0;
}
