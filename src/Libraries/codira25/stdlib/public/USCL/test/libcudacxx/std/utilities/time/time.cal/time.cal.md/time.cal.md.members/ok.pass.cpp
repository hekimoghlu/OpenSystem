/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
// class month_day;

// constexpr bool ok() const noexcept;
//  Returns: true if m_.ok() is true, 1d <= d_, and d_ is less than or equal to the
//    number of days in month m_; otherwise returns false.
//  When m_ == February, the number of days is considered to be 29.

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using day       = cuda::std::chrono::day;
  using month     = cuda::std::chrono::month;
  using month_day = cuda::std::chrono::month_day;

  static_assert(noexcept(cuda::std::declval<const month_day>().ok()));
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::declval<const month_day>().ok())>);

  static_assert(!month_day{}.ok(), "");
  static_assert(month_day{cuda::std::chrono::May, day{2}}.ok(), "");

  assert(!(month_day(cuda::std::chrono::April, day{0}).ok()));

  assert((month_day{cuda::std::chrono::March, day{1}}.ok()));
  for (unsigned i = 1; i <= 12; ++i)
  {
    const bool is31 = i == 1 || i == 3 || i == 5 || i == 7 || i == 8 || i == 10 || i == 12;
    assert(!(month_day{month{i}, day{0}}.ok()));
    assert((month_day{month{i}, day{1}}.ok()));
    assert((month_day{month{i}, day{10}}.ok()));
    assert((month_day{month{i}, day{29}}.ok()));
    assert((month_day{month{i}, day{30}}.ok()) == (i != 2));
    assert((month_day{month{i}, day{31}}.ok()) == is31);
    assert(!(month_day{month{i}, day{32}}.ok()));
  }

  //  If the month is not ok, all the days are bad
  for (unsigned i = 1; i <= 35; ++i)
  {
    assert(!(month_day{month{13}, day{i}}.ok()));
  }

  return 0;
}
