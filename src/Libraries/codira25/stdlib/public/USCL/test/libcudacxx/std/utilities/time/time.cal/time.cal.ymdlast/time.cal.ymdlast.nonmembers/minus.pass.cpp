/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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

// constexpr year_month_day_last
//   operator-(const year_month_day_last& ymdl, const months& dm) noexcept;
//
//   Returns: ymdl + (-dm).
//
// constexpr year_month_day_last
//   operator-(const year_month_day_last& ymdl, const years& dy) noexcept;
//
//   Returns: ymdl + (-dy).

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool testConstexprYears(cuda::std::chrono::year_month_day_last ymdl)
{
  cuda::std::chrono::year_month_day_last ym1 = ymdl - cuda::std::chrono::years{10};
  return ym1.year() == cuda::std::chrono::year{static_cast<int>(ymdl.year()) - 10} && ym1.month() == ymdl.month();
}

__host__ __device__ constexpr bool testConstexprMonths(cuda::std::chrono::year_month_day_last ymdl)
{
  cuda::std::chrono::year_month_day_last ym1 = ymdl - cuda::std::chrono::months{6};
  return ym1.year() == ymdl.year() && ym1.month() == cuda::std::chrono::month{static_cast<unsigned>(ymdl.month()) - 6};
}

int main(int, char**)
{
  using year                = cuda::std::chrono::year;
  using month               = cuda::std::chrono::month;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;
  using months              = cuda::std::chrono::months;
  using years               = cuda::std::chrono::years;

  constexpr month December = cuda::std::chrono::December;

  { // year_month_day_last - years
    static_assert(noexcept(cuda::std::declval<year_month_day_last>() - cuda::std::declval<years>()));
    static_assert(
      cuda::std::is_same_v<year_month_day_last,
                           decltype(cuda::std::declval<year_month_day_last>() - cuda::std::declval<years>())>);

    static_assert(testConstexprYears(year_month_day_last{year{1234}, month_day_last{December}}), "");
    year_month_day_last ym{year{1234}, month_day_last{December}};
    for (int i = 0; i <= 10; ++i)
    {
      year_month_day_last ym1 = ym - years{i};
      assert(static_cast<int>(ym1.year()) == 1234 - i);
      assert(ym1.month() == December);
    }
  }

  { // year_month_day_last - months
    static_assert(noexcept(cuda::std::declval<year_month_day_last>() - cuda::std::declval<months>()));
    static_assert(
      cuda::std::is_same_v<year_month_day_last,
                           decltype(cuda::std::declval<year_month_day_last>() - cuda::std::declval<months>())>);

    static_assert(testConstexprMonths(year_month_day_last{year{1234}, month_day_last{December}}), "");
    //  TODO test wrapping
    year_month_day_last ym{year{1234}, month_day_last{December}};
    for (unsigned i = 0; i <= 10; ++i)
    {
      year_month_day_last ym1 = ym - months{i};
      assert(static_cast<int>(ym1.year()) == 1234);
      assert(static_cast<unsigned>(ym1.month()) == 12U - i);
    }
  }

  return 0;
}
