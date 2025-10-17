/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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
// constexpr chrono::seconds seconds() const noexcept;
//
// See the table in hours.pass.cpp for correspondence between the magic values used below

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

template <typename Duration>
__host__ __device__ constexpr auto check_seconds(Duration d)
{
  using HMS = cuda::std::chrono::hh_mm_ss<Duration>;
  static_assert(cuda::std::is_same_v<cuda::std::chrono::seconds, decltype(cuda::std::declval<HMS>().seconds())>);
  static_assert(noexcept(cuda::std::declval<HMS>().seconds()));
  return HMS(d).seconds().count();
}

int main(int, char**)
{
  using microfortnights = cuda::std::chrono::duration<int, cuda::std::ratio<756, 625>>;

  static_assert(check_seconds(cuda::std::chrono::seconds(1)) == 1, "");
  static_assert(check_seconds(cuda::std::chrono::seconds(-1)) == 1, "");

  assert(check_seconds(cuda::std::chrono::seconds(5000)) == 20);
  assert(check_seconds(cuda::std::chrono::seconds(-5000)) == 20);
  assert(check_seconds(cuda::std::chrono::minutes(5000)) == 0);
  assert(check_seconds(cuda::std::chrono::minutes(-5000)) == 0);
  assert(check_seconds(cuda::std::chrono::hours(11)) == 0);
  assert(check_seconds(cuda::std::chrono::hours(-11)) == 0);

  assert(check_seconds(cuda::std::chrono::milliseconds(123456789LL)) == 36);
  assert(check_seconds(cuda::std::chrono::milliseconds(-123456789LL)) == 36);
  assert(check_seconds(cuda::std::chrono::microseconds(123456789LL)) == 3);
  assert(check_seconds(cuda::std::chrono::microseconds(-123456789LL)) == 3);
  assert(check_seconds(cuda::std::chrono::nanoseconds(123456789LL)) == 0);
  assert(check_seconds(cuda::std::chrono::nanoseconds(-123456789LL)) == 0);

  assert(check_seconds(microfortnights(1000)) == 9);
  assert(check_seconds(microfortnights(-1000)) == 9);
  assert(check_seconds(microfortnights(10000)) == 36);
  assert(check_seconds(microfortnights(-10000)) == 36);

  return 0;
}
