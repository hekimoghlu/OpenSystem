/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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

// is_trivially_copyable for volatile qualified types

// These compilers have not implemented Core 2094 which makes volatile
// qualified types trivially copyable.
// XFAIL: gcc-7, gcc-8, gcc-9

// When we marked this XFAIL for MSVC, QA reported that it unexpectedly passed.
// When we stopped marking it XFAIL for MSVC, QA reported that it unexpectedly
// failed
// UNSUPPORTED: msvc

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_trivially_copyable_volatile()
{
  static_assert(cuda::std::is_trivially_copyable<volatile T>::value, "");
  static_assert(cuda::std::is_trivially_copyable<const volatile T>::value, "");
  static_assert(cuda::std::is_trivially_copyable_v<volatile T>, "");
  static_assert(cuda::std::is_trivially_copyable_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_trivially_copyable_volatile()
{
  static_assert(!cuda::std::is_trivially_copyable<volatile T>::value, "");
  static_assert(!cuda::std::is_trivially_copyable<const volatile T>::value, "");
  static_assert(!cuda::std::is_trivially_copyable_v<volatile T>, "");
  static_assert(!cuda::std::is_trivially_copyable_v<const volatile T>, "");
}

struct A
{
  int i_;
};

struct B
{
  int i_;
  __host__ __device__ ~B()
  {
    assert(i_ == 0);
  }
};

class C
{
public:
  __host__ __device__ C();
};

int main(int, char**)
{
  test_is_trivially_copyable_volatile<int>();
  test_is_trivially_copyable_volatile<const int>();
  test_is_trivially_copyable_volatile<A>();
  test_is_trivially_copyable_volatile<const A>();
  test_is_trivially_copyable_volatile<C>();

  test_is_not_trivially_copyable_volatile<int&>();
  test_is_not_trivially_copyable_volatile<const A&>();
  test_is_not_trivially_copyable_volatile<B>();

  return 0;
}
