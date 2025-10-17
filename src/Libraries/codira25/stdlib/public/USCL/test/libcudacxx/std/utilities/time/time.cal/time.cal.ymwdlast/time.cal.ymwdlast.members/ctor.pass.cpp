/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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
// class year_month_weekday_last;

//  constexpr year_month_weekday_last(const chrono::year& y, const chrono::month& m,
//                               const chrono::weekday_last& wdl) noexcept;
//
//  Effects:  Constructs an object of type year_month_weekday_last by initializing
//                y_ with y, m_ with m, and wdl_ with wdl.
//
//  constexpr chrono::year                 year() const noexcept;
//  constexpr chrono::month               month() const noexcept;
//  constexpr chrono::weekday           weekday() const noexcept;
//  constexpr chrono::weekday_last weekday_last() const noexcept;
//  constexpr bool                           ok() const noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                    = cuda::std::chrono::year;
  using month                   = cuda::std::chrono::month;
  using weekday                 = cuda::std::chrono::weekday;
  using weekday_last            = cuda::std::chrono::weekday_last;
  using year_month_weekday_last = cuda::std::chrono::year_month_weekday_last;

  constexpr month January   = cuda::std::chrono::January;
  constexpr weekday Tuesday = cuda::std::chrono::Tuesday;

  static_assert(noexcept(year_month_weekday_last{year{1}, month{1}, weekday_last{Tuesday}}));

  constexpr year_month_weekday_last ym1{year{2019}, January, weekday_last{Tuesday}};
  static_assert(ym1.year() == year{2019}, "");
  static_assert(ym1.month() == January, "");
  static_assert(ym1.weekday() == Tuesday, "");
  static_assert(ym1.weekday_last() == weekday_last{Tuesday}, "");
  static_assert(ym1.ok(), "");

  return 0;
}
