/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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

//  year_month_day() = default;
//  constexpr year_month_day(const chrono::year& y, const chrono::month& m,
//                                   const chrono::day& d) noexcept;
//
//  Effects:  Constructs an object of type year_month_day by initializing
//                y_ with y, m_ with m, and d_ with d.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr chrono::day     day() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year           = cuda::std::chrono::year;
  using month          = cuda::std::chrono::month;
  using day            = cuda::std::chrono::day;
  using year_month_day = cuda::std::chrono::year_month_day;

  static_assert(noexcept(year_month_day{}));
  static_assert(noexcept(year_month_day{year{1}, month{1}, day{1}}));

  constexpr month January = cuda::std::chrono::January;

  constexpr year_month_day ym0{};
  static_assert(ym0.year() == year{}, "");
  static_assert(ym0.month() == month{}, "");
  static_assert(ym0.day() == day{}, "");
  static_assert(!ym0.ok(), "");

  constexpr year_month_day ym1{year{2019}, January, day{12}};
  static_assert(ym1.year() == year{2019}, "");
  static_assert(ym1.month() == January, "");
  static_assert(ym1.day() == day{12}, "");
  static_assert(ym1.ok(), "");

  return 0;
}
