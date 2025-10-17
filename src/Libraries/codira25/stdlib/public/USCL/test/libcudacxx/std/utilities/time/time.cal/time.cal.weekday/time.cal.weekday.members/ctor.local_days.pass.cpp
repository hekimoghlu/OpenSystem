/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
// class weekday;

//  constexpr weekday(const local_days& dp) noexcept;
//
//  Effects:  Constructs an object of type weekday by computing what day
//              of the week  corresponds to the local_days dp, and representing
//              that day of the week in wd_
//
//  Remarks: For any value ymd of type year_month_day for which ymd.ok() is true,
//                ymd == year_month_day{sys_days{ymd}} is true.
//
// [Example:
//  If dp represents 1970-01-01, the constructed weekday represents Thursday by storing 4 in wd_.
// â€”end example]

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using local_days = cuda::std::chrono::local_days;
  using days       = cuda::std::chrono::days;
  using weekday    = cuda::std::chrono::weekday;

  static_assert(noexcept(weekday{cuda::std::declval<local_days>()}));

  {
    constexpr local_days sd{}; // 1-Jan-1970 was a Thursday
    constexpr weekday wd{sd};

    static_assert(wd.ok(), "");
    static_assert(wd.c_encoding() == 4, "");
  }

  {
    constexpr local_days sd{days{10957 + 32}}; // 2-Feb-2000 was a Wednesday
    constexpr weekday wd{sd};

    static_assert(wd.ok(), "");
    static_assert(wd.c_encoding() == 3, "");
  }

  {
    constexpr local_days sd{days{-10957}}; // 2-Jan-1940 was a Tuesday
    constexpr weekday wd{sd};

    static_assert(wd.ok(), "");
    static_assert(wd.c_encoding() == 2, "");
  }

  {
    local_days sd{days{-(10957 + 34)}}; // 29-Nov-1939 was a Wednesday
    weekday wd{sd};

    assert(wd.ok());
    assert(wd.c_encoding() == 3);
  }

  return 0;
}
