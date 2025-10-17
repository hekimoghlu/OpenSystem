/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
//
// constexpr chrono::hours hours() const noexcept;

// Test values
// duration     hours   minutes seconds fractional
// 5000s            1       23      20      0
// 5000 minutes     83      20      0       0
// 123456789ms      34      17      36      789ms
// 123456789us      0       2       3       456789us
// 123456789ns      0       0       0       123456789ns
// 1000mfn          0       20      9       0.6 (6000/10000)
// 10000mfn         3       21      36      0

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

template <typename Duration>
__host__ __device__ constexpr long check_hours(Duration d)
{
  using HMS = cuda::std::chrono::hh_mm_ss<Duration>;
  static_assert(cuda::std::is_same_v<cuda::std::chrono::hours, decltype(cuda::std::declval<HMS>().hours())>);
  static_assert(noexcept(cuda::std::declval<HMS>().hours()));
  return HMS(d).hours().count();
}

int main(int, char**)
{
  using microfortnights = cuda::std::chrono::duration<int, cuda::std::ratio<756, 625>>;

  static_assert(check_hours(cuda::std::chrono::minutes(1)) == 0, "");
  static_assert(check_hours(cuda::std::chrono::minutes(-1)) == 0, "");

  assert(check_hours(cuda::std::chrono::seconds(5000)) == 1);
  assert(check_hours(cuda::std::chrono::seconds(-5000)) == 1);
  assert(check_hours(cuda::std::chrono::minutes(5000)) == 83);
  assert(check_hours(cuda::std::chrono::minutes(-5000)) == 83);
  assert(check_hours(cuda::std::chrono::hours(11)) == 11);
  assert(check_hours(cuda::std::chrono::hours(-11)) == 11);

  assert(check_hours(cuda::std::chrono::milliseconds(123456789LL)) == 34);
  assert(check_hours(cuda::std::chrono::milliseconds(-123456789LL)) == 34);
  assert(check_hours(cuda::std::chrono::microseconds(123456789LL)) == 0);
  assert(check_hours(cuda::std::chrono::microseconds(-123456789LL)) == 0);
  assert(check_hours(cuda::std::chrono::nanoseconds(123456789LL)) == 0);
  assert(check_hours(cuda::std::chrono::nanoseconds(-123456789LL)) == 0);

  assert(check_hours(microfortnights(1000)) == 0);
  assert(check_hours(microfortnights(-1000)) == 0);
  assert(check_hours(microfortnights(10000)) == 3);
  assert(check_hours(microfortnights(-10000)) == 3);

  return 0;
}
