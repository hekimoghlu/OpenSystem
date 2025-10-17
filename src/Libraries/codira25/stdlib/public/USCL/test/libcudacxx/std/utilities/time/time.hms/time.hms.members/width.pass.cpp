/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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

// <chrono>

// template <class Duration>
// class hh_mm_ss
// {
// public:
//     static unsigned constexpr fractional_width = see below;
//     using precision                            = see below;
//
//	fractional_width is the number of fractional decimal digits represented by precision.
//  fractional_width has the value of the smallest possible integer in the range [0, 18]
//    such that precision will exactly represent all values of Duration.
//  If no such value of fractional_width exists, then fractional_width is 6.

#include <uscl/std/chrono>

#include "test_macros.h"

template <typename Duration, unsigned width>
__host__ __device__ constexpr bool check_width()
{
  using HMS = cuda::std::chrono::hh_mm_ss<Duration>;
  return HMS::fractional_width == width;
}

int main(int, char**)
{
  using microfortnights = cuda::std::chrono::duration<int, cuda::std::ratio<756, 625>>;

  static_assert(check_width<cuda::std::chrono::hours, 0>(), "");
  static_assert(check_width<cuda::std::chrono::minutes, 0>(), "");
  static_assert(check_width<cuda::std::chrono::seconds, 0>(), "");
  static_assert(check_width<cuda::std::chrono::milliseconds, 3>(), "");
  static_assert(check_width<cuda::std::chrono::microseconds, 6>(), "");
  static_assert(check_width<cuda::std::chrono::nanoseconds, 9>(), "");
  static_assert(check_width<cuda::std::chrono::duration<int, cuda::std::ratio<1, 2>>, 1>(), "");
  static_assert(check_width<cuda::std::chrono::duration<int, cuda::std::ratio<1, 3>>, 6>(), "");
  static_assert(check_width<cuda::std::chrono::duration<int, cuda::std::ratio<1, 4>>, 2>(), "");
  static_assert(check_width<cuda::std::chrono::duration<int, cuda::std::ratio<1, 5>>, 1>(), "");
  static_assert(check_width<cuda::std::chrono::duration<int, cuda::std::ratio<1, 6>>, 6>(), "");
  static_assert(check_width<cuda::std::chrono::duration<int, cuda::std::ratio<1, 7>>, 6>(), "");
  static_assert(check_width<cuda::std::chrono::duration<int, cuda::std::ratio<1, 8>>, 3>(), "");
  static_assert(check_width<cuda::std::chrono::duration<int, cuda::std::ratio<1, 9>>, 6>(), "");
  static_assert(check_width<cuda::std::chrono::duration<int, cuda::std::ratio<1, 10>>, 1>(), "");
  static_assert(check_width<microfortnights, 4>(), "");

  return 0;
}
