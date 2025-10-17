/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
// class year_month_day;

// constexpr bool operator==(const year_month_day& x, const year_month_day& y) noexcept;
//   Returns: x.year() == y.year() && x.month() == y.month().
//
// constexpr bool operator< (const year_month_day& x, const year_month_day& y) noexcept;
//   Returns:
//      If x.year() < y.year() returns true.
//      Otherwise, if x.year() > y.year() returns false.
//      Otherwise, if x.month() < y.month() returns true.
//      Otherwise, if x.month() > y.month() returns false.
//      Otherwise, returns x.day() < y.day()

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using day            = cuda::std::chrono::day;
  using year           = cuda::std::chrono::year;
  using month          = cuda::std::chrono::month;
  using year_month_day = cuda::std::chrono::year_month_day;

  AssertComparisonsAreNoexcept<year_month_day>();
  AssertComparisonsReturnBool<year_month_day>();

  constexpr month January  = cuda::std::chrono::January;
  constexpr month February = cuda::std::chrono::February;

  static_assert(
    testComparisons(
      year_month_day{year{1234}, January, day{1}}, year_month_day{year{1234}, January, day{1}}, true, false),
    "");

  //  different day
  static_assert(
    testComparisons(
      year_month_day{year{1234}, January, day{1}}, year_month_day{year{1234}, January, day{2}}, false, true),
    "");

  //  different month
  static_assert(
    testComparisons(
      year_month_day{year{1234}, January, day{1}}, year_month_day{year{1234}, February, day{1}}, false, true),
    "");

  //  different year
  static_assert(
    testComparisons(
      year_month_day{year{1234}, January, day{1}}, year_month_day{year{1235}, January, day{1}}, false, true),
    "");

  //  different month and day
  static_assert(
    testComparisons(
      year_month_day{year{1234}, January, day{2}}, year_month_day{year{1234}, February, day{1}}, false, true),
    "");

  //  different year and month
  static_assert(
    testComparisons(
      year_month_day{year{1234}, February, day{1}}, year_month_day{year{1235}, January, day{1}}, false, true),
    "");

  //  different year and day
  static_assert(
    testComparisons(
      year_month_day{year{1234}, January, day{2}}, year_month_day{year{1235}, January, day{1}}, false, true),
    "");

  //  different year, month and day
  static_assert(
    testComparisons(
      year_month_day{year{1234}, February, day{2}}, year_month_day{year{1235}, January, day{1}}, false, true),
    "");

  //  same year, different days
  for (unsigned i = 1; i < 28; ++i)
  {
    for (unsigned j = 1; j < 28; ++j)
    {
      assert((testComparisons(
        year_month_day{year{1234}, January, day{i}}, year_month_day{year{1234}, January, day{j}}, i == j, i < j)));
    }
  }

  //  same year, different months
  for (unsigned i = 1; i < 12; ++i)
  {
    for (unsigned j = 1; j < 12; ++j)
    {
      assert((testComparisons(
        year_month_day{year{1234}, month{i}, day{12}}, year_month_day{year{1234}, month{j}, day{12}}, i == j, i < j)));
    }
  }

  //  same month, different years
  for (int i = 1000; i < 20; ++i)
  {
    for (int j = 1000; j < 20; ++j)
    {
      assert((testComparisons(
        year_month_day{year{i}, January, day{12}}, year_month_day{year{j}, January, day{12}}, i == j, i < j)));
    }
  }

  return 0;
}
