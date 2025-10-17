/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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

// is_nothrow_copy_assignable

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_has_nothrow_assign()
{
  static_assert(cuda::std::is_nothrow_copy_assignable<T>::value, "");
  static_assert(cuda::std::is_nothrow_copy_assignable_v<T>, "");
}

template <class T>
__host__ __device__ void test_has_not_nothrow_assign()
{
  static_assert(!cuda::std::is_nothrow_copy_assignable<T>::value, "");
  static_assert(!cuda::std::is_nothrow_copy_assignable_v<T>, "");
}

class Empty
{};

struct NotEmpty
{
  __host__ __device__ virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

struct A
{
  __host__ __device__ A& operator=(const A&);
};

int main(int, char**)
{
  test_has_nothrow_assign<int&>();
  test_has_nothrow_assign<Union>();
  test_has_nothrow_assign<Empty>();
  test_has_nothrow_assign<int>();
  test_has_nothrow_assign<double>();
  test_has_nothrow_assign<int*>();
  test_has_nothrow_assign<const int*>();
  test_has_nothrow_assign<NotEmpty>();
  test_has_nothrow_assign<bit_zero>();

  test_has_not_nothrow_assign<const int>();
  test_has_not_nothrow_assign<void>();
#if !TEST_COMPILER(NVHPC)
  test_has_not_nothrow_assign<A>();
#endif // !TEST_COMPILER(NVHPC)

  return 0;
}
