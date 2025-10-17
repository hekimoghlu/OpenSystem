/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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

// alignment_of

#include <uscl/std/cstdint>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T, unsigned A>
__host__ __device__ void test_alignment_of()
{
  const unsigned AlignofResult = alignof(T);
  static_assert(AlignofResult == A, "Golden value does not match result of alignof keyword");
  static_assert(cuda::std::alignment_of<T>::value == AlignofResult, "");
  static_assert(cuda::std::alignment_of<T>::value == A, "");
  static_assert(cuda::std::alignment_of<const T>::value == A, "");
  static_assert(cuda::std::alignment_of<volatile T>::value == A, "");
  static_assert(cuda::std::alignment_of<const volatile T>::value == A, "");
  static_assert(cuda::std::alignment_of_v<T> == A, "");
  static_assert(cuda::std::alignment_of_v<const T> == A, "");
  static_assert(cuda::std::alignment_of_v<volatile T> == A, "");
  static_assert(cuda::std::alignment_of_v<const volatile T> == A, "");
}

class Class
{
public:
  __host__ __device__ ~Class();
};

int main(int, char**)
{
  test_alignment_of<int&, 4>();
  test_alignment_of<Class, 1>();
  test_alignment_of<int*, sizeof(intptr_t)>();
  test_alignment_of<const int*, sizeof(intptr_t)>();
  test_alignment_of<char[3], 1>();
  test_alignment_of<int, 4>();
  // The test case below is a hack. It's hard to detect what golden value
  // we should expect. In most cases it should be 8. But in i386 builds
  // with Clang >= 8 or GCC >= 8 the value is '4'.
  test_alignment_of<double, alignof(double)>();
  test_alignment_of<bool, 1>();
  test_alignment_of<unsigned, 4>();

  return 0;
}
