/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
// class year_month_weekday;

//  explicit constexpr year_month_weekday(const local_days& dp) noexcept;
//
//
//  Effects:  Constructs an object of type year_month_weekday that corresponds
//                to the date represented by dp
//
//  Remarks: Equivalent to constructing with sys_days{dp.time_since_epoch()}.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr chrono::day     day() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4307) // potential overflow
TEST_DIAG_SUPPRESS_MSVC(4308) // unsigned/signed comparisons

int main(int, char**)
{
  using year               = cuda::std::chrono::year;
  using days               = cuda::std::chrono::days;
  using local_days         = cuda::std::chrono::local_days;
  using weekday_indexed    = cuda::std::chrono::weekday_indexed;
  using year_month_weekday = cuda::std::chrono::year_month_weekday;

  static_assert(noexcept(year_month_weekday{cuda::std::declval<const local_days>()}));

  auto constexpr January = cuda::std::chrono::January;

  {
    constexpr local_days sd{}; // 1-Jan-1970 was a Thursday
    constexpr year_month_weekday ymwd{sd};
    auto constexpr Thursday = cuda::std::chrono::Thursday;

    static_assert(ymwd.ok(), "");
    static_assert(ymwd.year() == year{1970}, "");
    static_assert(ymwd.month() == January, "");
    static_assert(ymwd.weekday() == Thursday, "");
    static_assert(ymwd.index() == 1, "");
    static_assert(ymwd.weekday_indexed() == weekday_indexed{Thursday, 1}, "");
    static_assert(ymwd == year_month_weekday{local_days{ymwd}}, ""); // round trip
  }

  {
    constexpr local_days sd{days{10957 + 32}}; // 2-Feb-2000 was a Wednesday
    constexpr year_month_weekday ymwd{sd};

    auto constexpr February  = cuda::std::chrono::February;
    auto constexpr Wednesday = cuda::std::chrono::Wednesday;

    static_assert(ymwd.ok(), "");
    static_assert(ymwd.year() == year{2000}, "");
    static_assert(ymwd.month() == February, "");
    static_assert(ymwd.weekday() == Wednesday, "");
    static_assert(ymwd.index() == 1, "");
    static_assert(ymwd.weekday_indexed() == weekday_indexed{Wednesday, 1}, "");
    static_assert(ymwd == year_month_weekday{local_days{ymwd}}, ""); // round trip
  }

  {
    constexpr local_days sd{days{-10957}}; // 2-Jan-1940 was a Tuesday
    constexpr year_month_weekday ymwd{sd};

    auto constexpr Tuesday = cuda::std::chrono::Tuesday;

    static_assert(ymwd.ok(), "");
    static_assert(ymwd.year() == year{1940}, "");
    static_assert(ymwd.month() == January, "");
    static_assert(ymwd.weekday() == Tuesday, "");
    static_assert(ymwd.index() == 1, "");
    static_assert(ymwd.weekday_indexed() == weekday_indexed{Tuesday, 1}, "");
    static_assert(ymwd == year_month_weekday{local_days{ymwd}}, ""); // round trip
  }

  {
    local_days sd{days{-(10957 + 34)}}; // 29-Nov-1939 was a Wednesday
    year_month_weekday ymwd{sd};
    auto constexpr November  = cuda::std::chrono::November;
    auto constexpr Wednesday = cuda::std::chrono::Wednesday;

    assert(ymwd.ok());
    assert(ymwd.year() == year{1939});
    assert(ymwd.month() == November);
    assert(ymwd.weekday() == Wednesday);
    assert(ymwd.index() == 5);
    assert((ymwd.weekday_indexed() == weekday_indexed{Wednesday, 5}));
    assert(ymwd == year_month_weekday{local_days{ymwd}}); // round trip
  }

  return 0;
}
