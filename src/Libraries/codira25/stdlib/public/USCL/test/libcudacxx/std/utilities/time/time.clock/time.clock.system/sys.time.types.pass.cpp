/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
// UNSUPPORTED: c++17

// <cuda/std/chrono>

// template<class Duration>
//   using sys_time  = time_point<system_clock, Duration>;
// using sys_seconds = sys_time<seconds>;
// using sys_days    = sys_time<days>;

// [Example:
//   sys_seconds{sys_days{1970y/January/1}}.time_since_epoch() is 0s.
//   sys_seconds{sys_days{2000y/January/1}}.time_since_epoch() is 946â€™684â€™800s, which is 10â€™957 * 86â€™400s.
// â€”end example]

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  using system_clock = cuda::std::chrono::system_clock;
  using year         = cuda::std::chrono::year;

  using seconds = cuda::std::chrono::seconds;
  using minutes = cuda::std::chrono::minutes;
  using days    = cuda::std::chrono::days;

  using sys_seconds = cuda::std::chrono::sys_seconds;
  using sys_minutes = cuda::std::chrono::sys_time<minutes>;
  using sys_days    = cuda::std::chrono::sys_days;

  constexpr cuda::std::chrono::month January = cuda::std::chrono::January;

  static_assert(cuda::std::is_same_v<cuda::std::chrono::sys_time<seconds>, sys_seconds>);
  static_assert(cuda::std::is_same_v<cuda::std::chrono::sys_time<days>, sys_days>);

  //  Test the long form, too
  static_assert(cuda::std::is_same_v<cuda::std::chrono::time_point<system_clock, seconds>, sys_seconds>);
  static_assert(cuda::std::is_same_v<cuda::std::chrono::time_point<system_clock, minutes>, sys_minutes>);
  static_assert(cuda::std::is_same_v<cuda::std::chrono::time_point<system_clock, days>, sys_days>);

  //  Test some well known values
  sys_days d0 = sys_days{year{1970} / January / 1};
  sys_days d1 = sys_days{year{2000} / January / 1};
  static_assert(cuda::std::is_same_v<decltype(d0.time_since_epoch()), days>);
  assert(d0.time_since_epoch().count() == 0);
  assert(d1.time_since_epoch().count() == 10957);

  sys_seconds s0{d0};
  sys_seconds s1{d1};
  static_assert(cuda::std::is_same_v<decltype(s0.time_since_epoch()), seconds>);
  assert(s0.time_since_epoch().count() == 0);
  assert(s1.time_since_epoch().count() == 946684800L);

  return 0;
}
