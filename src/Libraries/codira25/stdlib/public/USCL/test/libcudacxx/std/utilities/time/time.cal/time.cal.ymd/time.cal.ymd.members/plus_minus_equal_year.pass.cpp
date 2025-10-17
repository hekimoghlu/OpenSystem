/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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

// constexpr year_month_day& operator+=(const years& d) noexcept;
// constexpr year_month_day& operator-=(const years& d) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <typename D, typename Ds>
__host__ __device__ constexpr bool testConstexpr(D d1)
{
  if (static_cast<int>((d1).year()) != 1)
  {
    return false;
  }
  if (static_cast<int>((d1 += Ds{1}).year()) != 2)
  {
    return false;
  }
  if (static_cast<int>((d1 += Ds{2}).year()) != 4)
  {
    return false;
  }
  if (static_cast<int>((d1 += Ds{12}).year()) != 16)
  {
    return false;
  }
  if (static_cast<int>((d1 -= Ds{1}).year()) != 15)
  {
    return false;
  }
  if (static_cast<int>((d1 -= Ds{2}).year()) != 13)
  {
    return false;
  }
  if (static_cast<int>((d1 -= Ds{12}).year()) != 1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year           = cuda::std::chrono::year;
  using month          = cuda::std::chrono::month;
  using day            = cuda::std::chrono::day;
  using year_month_day = cuda::std::chrono::year_month_day;
  using years          = cuda::std::chrono::years;

  static_assert(noexcept(cuda::std::declval<year_month_day&>() += cuda::std::declval<years>()));
  static_assert(noexcept(cuda::std::declval<year_month_day&>() -= cuda::std::declval<years>()));

  static_assert(cuda::std::is_same_v<year_month_day&,
                                     decltype(cuda::std::declval<year_month_day&>() += cuda::std::declval<years>())>);
  static_assert(cuda::std::is_same_v<year_month_day&,
                                     decltype(cuda::std::declval<year_month_day&>() -= cuda::std::declval<years>())>);

  static_assert(testConstexpr<year_month_day, years>(year_month_day{year{1}, month{1}, day{1}}), "");

  for (int i = 1000; i <= 1010; ++i)
  {
    month m{2};
    day d{23};
    year_month_day ym(year{i}, m, d);
    assert(static_cast<int>((ym += years{2}).year()) == i + 2);
    assert(ym.month() == m);
    assert(ym.day() == d);
    assert(static_cast<int>((ym).year()) == i + 2);
    assert(ym.month() == m);
    assert(ym.day() == d);
    assert(static_cast<int>((ym -= years{1}).year()) == i + 1);
    assert(ym.month() == m);
    assert(ym.day() == d);
    assert(static_cast<int>((ym).year()) == i + 1);
    assert(ym.month() == m);
    assert(ym.day() == d);
  }

  return 0;
}
