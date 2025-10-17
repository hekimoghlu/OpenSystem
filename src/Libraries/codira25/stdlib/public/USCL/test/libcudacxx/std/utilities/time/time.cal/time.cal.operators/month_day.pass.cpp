/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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

// constexpr month_day
//   operator/(const month& m, const day& d) noexcept;
// Returns: {m, d}.
//
// constexpr month_day
//   operator/(const day& d, const month& m) noexcept;
// Returns: m / d.

// constexpr month_day
//   operator/(const month& m, int d) noexcept;
// Returns: m / day(d).
//
// constexpr month_day
//   operator/(int m, const day& d) noexcept;
// Returns: month(m) / d.
//
// constexpr month_day
//   operator/(const day& d, int m) noexcept;
// Returns: month(m) / d.

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month_day = cuda::std::chrono::month_day;
  using month     = cuda::std::chrono::month;
  using day       = cuda::std::chrono::day;

  constexpr month February = cuda::std::chrono::February;

  { // operator/(const month& m, const day& d) (and switched)
    static_assert(noexcept(February / day{1}));
    static_assert(cuda::std::is_same_v<month_day, decltype(February / day{1})>);
    static_assert(noexcept(day{1} / February));
    static_assert(cuda::std::is_same_v<month_day, decltype(day{1} / February)>);

    for (int i = 1; i <= 12; ++i)
    {
      for (unsigned j = 0; j <= 30; ++j)
      {
        month m(i);
        day d{j};
        month_day md1 = m / d;
        month_day md2 = d / m;
        assert(md1.month() == m);
        assert(md1.day() == d);
        assert(md2.month() == m);
        assert(md2.day() == d);
        assert(md1 == md2);
      }
    }
  }

  { // operator/(const month& m, int d) (NOT switched)
    static_assert(noexcept(February / 2));
    static_assert(cuda::std::is_same_v<month_day, decltype(February / 2)>);

    for (int i = 1; i <= 12; ++i)
    {
      for (unsigned j = 0; j <= 30; ++j)
      {
        month m(i);
        day d(j);
        month_day md1 = m / j;
        assert(md1.month() == m);
        assert(md1.day() == d);
      }
    }
  }

  { // operator/(const day& d, int m) (and switched)
    static_assert(noexcept(day{2} / 2));
    static_assert(cuda::std::is_same_v<month_day, decltype(day{2} / 2)>);
    static_assert(noexcept(2 / day{2}));
    static_assert(cuda::std::is_same_v<month_day, decltype(2 / day{2})>);

    for (int i = 1; i <= 12; ++i)
    {
      for (unsigned j = 0; j <= 30; ++j)
      {
        month m(i);
        day d(j);
        month_day md1 = d / i;
        month_day md2 = i / d;
        assert(md1.month() == m);
        assert(md1.day() == d);
        assert(md2.month() == m);
        assert(md2.day() == d);
        assert(md1 == md2);
      }
    }
  }

  return 0;
}
