/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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
// class year_month_day_last;

// constexpr bool ok() const noexcept;
//  Returns: m_.ok() && y_.ok().

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                = cuda::std::chrono::year;
  using month               = cuda::std::chrono::month;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;

  constexpr month January = cuda::std::chrono::January;

  static_assert(noexcept(cuda::std::declval<const year_month_day_last>().ok()));
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::declval<const year_month_day_last>().ok())>);

  static_assert(!year_month_day_last{year{-32768}, month_day_last{month{}}}.ok(), ""); // both bad
  static_assert(!year_month_day_last{year{-32768}, month_day_last{January}}.ok(), ""); // Bad year
  static_assert(!year_month_day_last{year{2019}, month_day_last{month{}}}.ok(), ""); // Bad month
  static_assert(year_month_day_last{year{2019}, month_day_last{January}}.ok(), ""); // All OK

  for (unsigned i = 0; i <= 50; ++i)
  {
    year_month_day_last ym{year{2019}, month_day_last{month{i}}};
    assert(ym.ok() == month{i}.ok());
  }

  const int ymax = static_cast<int>(year::max());
  for (int i = ymax - 100; i <= ymax + 100; ++i)
  {
    year_month_day_last ym{year{i}, month_day_last{January}};
    assert(ym.ok() == year{i}.ok());
  }

  return 0;
}
