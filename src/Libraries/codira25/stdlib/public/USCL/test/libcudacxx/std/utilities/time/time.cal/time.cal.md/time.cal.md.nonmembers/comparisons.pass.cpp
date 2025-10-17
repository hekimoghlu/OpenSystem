/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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

// UNSUPPORTED: msvc-19.16

// <chrono>
// class month_day;

// constexpr bool operator==(const month_day& x, const month_day& y) noexcept;
//   Returns: x.month() == y.month() && x.day() == y.day().
//
// constexpr bool operator< (const month_day& x, const month_day& y) noexcept;
//   Returns:
//      If x.month() < y.month() returns true.
//      Otherwise, if x.month() > y.month() returns false.
//      Otherwise, returns x.day() < y.day().

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using day       = cuda::std::chrono::day;
  using month     = cuda::std::chrono::month;
  using month_day = cuda::std::chrono::month_day;

  AssertComparisonsAreNoexcept<month_day>();
  AssertComparisonsReturnBool<month_day>();

  static_assert(
    testComparisons(
      month_day{cuda::std::chrono::January, day{1}}, month_day{cuda::std::chrono::January, day{1}}, true, false),
    "");

  static_assert(
    testComparisons(
      month_day{cuda::std::chrono::January, day{1}}, month_day{cuda::std::chrono::January, day{2}}, false, true),
    "");

  static_assert(
    testComparisons(
      month_day{cuda::std::chrono::January, day{1}}, month_day{cuda::std::chrono::February, day{1}}, false, true),
    "");

  //  same day, different months
  for (unsigned i = 1; i < 12; ++i)
  {
    for (unsigned j = 1; j < 12; ++j)
    {
      assert((testComparisons(month_day{month{i}, day{1}}, month_day{month{j}, day{1}}, i == j, i < j)));
    }
  }

  //  same month, different days
  for (unsigned i = 1; i < 31; ++i)
  {
    for (unsigned j = 1; j < 31; ++j)
    {
      assert((testComparisons(month_day{month{2}, day{i}}, month_day{month{2}, day{j}}, i == j, i < j)));
    }
  }

  return 0;
}
