/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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

// is_nothrow_assignable

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_is_nothrow_assignable()
{
  static_assert((cuda::std::is_nothrow_assignable<T, U>::value), "");
  static_assert((cuda::std::is_nothrow_assignable_v<T, U>), "");
}

template <class T, class U>
__host__ __device__ void test_is_not_nothrow_assignable()
{
  static_assert((!cuda::std::is_nothrow_assignable<T, U>::value), "");
  static_assert((!cuda::std::is_nothrow_assignable_v<T, U>), "");
}

struct A
{};

struct B
{
  __host__ __device__ void operator=(A);
};

struct C
{
  __host__ __device__ void operator=(C&); // not const
};

int main(int, char**)
{
  test_is_nothrow_assignable<int&, int&>();
  test_is_nothrow_assignable<int&, int>();
#if !defined(_LIBCUDACXX_HAS_NOEXCEPT_SFINAE)
  // The `__has_nothrow_assign`-based fallback for can't handle this case.
  test_is_nothrow_assignable<int&, double>();
#endif

  test_is_not_nothrow_assignable<int, int&>();
  test_is_not_nothrow_assignable<int, int>();

  test_is_not_nothrow_assignable<A, B>();
#if !TEST_COMPILER(NVHPC)
  test_is_not_nothrow_assignable<B, A>();
  test_is_not_nothrow_assignable<C, C&>();
#endif // !TEST_COMPILER(NVHPC)

  return 0;
}
