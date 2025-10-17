/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
// constexpr precision to_duration() const noexcept;
//
// See the table in hours.pass.cpp for correspondence between the magic values used below

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

template <typename Duration>
__host__ __device__ constexpr long long check_duration(Duration d)
{
  using HMS = cuda::std::chrono::hh_mm_ss<Duration>;
  static_assert(cuda::std::is_same_v<typename HMS::precision, decltype(cuda::std::declval<HMS>().to_duration())>);
  static_assert(noexcept(cuda::std::declval<HMS>().to_duration()));

  return HMS(d).to_duration().count();
}

int main(int, char**)
{
  using microfortnights = cuda::std::chrono::duration<int, cuda::std::ratio<756, 625>>;

  static_assert(check_duration(cuda::std::chrono::minutes(1)) == 60, "");
  static_assert(check_duration(cuda::std::chrono::minutes(-1)) == -60, "");

  assert(check_duration(cuda::std::chrono::seconds(5000)) == 5000LL);
  assert(check_duration(cuda::std::chrono::seconds(-5000)) == -5000LL);
  assert(check_duration(cuda::std::chrono::minutes(5000)) == 300000LL);
  assert(check_duration(cuda::std::chrono::minutes(-5000)) == -300000LL);
  assert(check_duration(cuda::std::chrono::hours(11)) == 39600LL);
  assert(check_duration(cuda::std::chrono::hours(-11)) == -39600LL);

  assert(check_duration(cuda::std::chrono::milliseconds(123456789LL)) == 123456789LL);
  assert(check_duration(cuda::std::chrono::milliseconds(-123456789LL)) == -123456789LL);
  assert(check_duration(cuda::std::chrono::microseconds(123456789LL)) == 123456789LL);
  assert(check_duration(cuda::std::chrono::microseconds(-123456789LL)) == -123456789LL);
  assert(check_duration(cuda::std::chrono::nanoseconds(123456789LL)) == 123456789LL);
  assert(check_duration(cuda::std::chrono::nanoseconds(-123456789LL)) == -123456789LL);

  assert(check_duration(microfortnights(1000)) == 12096000);
  assert(check_duration(microfortnights(-1000)) == -12096000);
  assert(check_duration(microfortnights(10000)) == 120960000);
  assert(check_duration(microfortnights(-10000)) == -120960000);

  return 0;
}
