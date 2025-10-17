/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

// UNSUPPORTED: clang && (!nvcc)
// XFAIL: *

// <chrono>
// class year_month;

// constexpr year_month operator-(const year_month& ym, const years& dy) noexcept;
// Returns: ym + -dy.
//
// constexpr year_month operator-(const year_month& ym, const months& dm) noexcept;
// Returns: ym + -dm.
//
// constexpr months operator-(const year_month& x, const year_month& y) noexcept;
// Returns: x.year() - y.year() + months{static_cast<int>(unsigned{x.month()}) -
//                                       static_cast<int>(unsigned{y.month()})}

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
  using year       = cuda::std::chrono::year;
  using years      = cuda::std::chrono::years;
  using month      = cuda::std::chrono::month;
  using months     = cuda::std::chrono::months;
  using year_month = cuda::std::chrono::year_month;

  auto constexpr January = cuda::std::chrono::January;

  { // year_month - years
    static_assert(noexcept(cuda::std::declval<year_month>() - cuda::std::declval<years>()));
    static_assert(
      cuda::std::is_same_v<year_month, decltype(cuda::std::declval<year_month>() - cuda::std::declval<years>())>);

    //  static_assert(testConstexprYears (year_month{year{1}, month{1}}), "");

    year_month ym{year{1234}, January};
    for (int i = 0; i <= 10; ++i)
    {
      year_month ym1 = ym - years{i};
      assert(static_cast<int>(ym1.year()) == 1234 - i);
      assert(ym1.month() == cuda::std::chrono::January);
    }
  }

  { // year_month - months
    static_assert(noexcept(cuda::std::declval<year_month>() - cuda::std::declval<months>()));
    static_assert(
      cuda::std::is_same_v<year_month, decltype(cuda::std::declval<year_month>() - cuda::std::declval<months>())>);

    //  static_assert(testConstexprMonths(year_month{year{1}, month{1}}), "");

    auto constexpr November = cuda::std::chrono::November;
    year_month ym{year{1234}, November};
    for (int i = 0; i <= 10; ++i) // TODO test wrap-around
    {
      year_month ym1 = ym - months{i};
      assert(static_cast<int>(ym1.year()) == 1234);
      assert(ym1.month() == month(11 - i));
    }
  }

  { // year_month - year_month
    static_assert(noexcept(cuda::std::declval<year_month>() - cuda::std::declval<year_month>()));
    static_assert(
      cuda::std::is_same_v<months, decltype(cuda::std::declval<year_month>() - cuda::std::declval<year_month>())>);

    //  static_assert(testConstexprMonths(year_month{year{1}, month{1}}), "");

    //  Same year
    year y{2345};
    for (int i = 1; i <= 12; ++i)
    {
      for (int j = 1; j <= 12; ++j)
      {
        months diff = year_month{y, month(i)} - year_month{y, month(j)};
        std::cout << "i: " << i << " j: " << j << " -> " << diff.count() << std::endl;
        assert(diff.count() == i - j);
      }
    }

    //  TODO: different year
  }

  return 0;
}
