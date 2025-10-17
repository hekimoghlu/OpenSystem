/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
// Due to C++17 inline variables ASAN flags this test as containing an ODR
// violation because Clock::is_steady is defined in both the dylib and this TU.
// UNSUPPORTED: asan

// <cuda/std/chrono>

// high_resolution_clock

// check clock invariants

#include <uscl/std/chrono>

template <class T>
__host__ __device__ void test(const T&)
{}

int main(int, char**)
{
  typedef cuda::std::chrono::high_resolution_clock C;
  static_assert((cuda::std::is_same<C::rep, C::duration::rep>::value), "");
  static_assert((cuda::std::is_same<C::period, C::duration::period>::value), "");
  static_assert((cuda::std::is_same<C::duration, C::time_point::duration>::value), "");
  static_assert(C::is_steady || !C::is_steady, "");
  test(+cuda::std::chrono::high_resolution_clock::is_steady);

  return 0;
}
