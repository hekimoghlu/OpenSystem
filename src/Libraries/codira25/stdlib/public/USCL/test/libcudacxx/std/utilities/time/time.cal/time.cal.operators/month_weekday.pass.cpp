/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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
// class month_weekday;

// constexpr month_weekday
//   operator/(const month& m, const weekday_indexed& wdi) noexcept;
// Returns: {m, wdi}.
//
// constexpr month_weekday
//   operator/(int m, const weekday_indexed& wdi) noexcept;
// Returns: month(m) / wdi.
//
// constexpr month_weekday
//   operator/(const weekday_indexed& wdi, const month& m) noexcept;
// Returns: m / wdi. constexpr month_weekday
//
// constexpr month_weekday
//   operator/(const weekday_indexed& wdi, int m) noexcept;
// Returns: month(m) / wdi.

//
// [Example:
// constexpr auto mwd = February/Tuesday[3]; // mwd is the third Tuesday of February of an as yet unspecified year
//      static_assert(mwd.month() == February);
//      static_assert(mwd.weekday_indexed() == Tuesday[3]);
// â€”end example]

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month_weekday   = cuda::std::chrono::month_weekday;
  using month           = cuda::std::chrono::month;
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  constexpr weekday Tuesday = cuda::std::chrono::Tuesday;
  constexpr month February  = cuda::std::chrono::February;

  { // operator/(const month& m, const weekday_indexed& wdi) (and switched)
    static_assert(noexcept(February / Tuesday[2]));
    static_assert(cuda::std::is_same_v<month_weekday, decltype(February / Tuesday[2])>);
    static_assert(noexcept(Tuesday[2] / February));
    static_assert(cuda::std::is_same_v<month_weekday, decltype(Tuesday[2] / February)>);

    //  Run the example
    {
      constexpr month_weekday wdi = February / Tuesday[3];
      static_assert(wdi.month() == February, "");
      static_assert(wdi.weekday_indexed() == Tuesday[3], "");
    }

    for (int i = 1; i <= 12; ++i)
    {
      for (unsigned j = 0; j <= 6; ++j)
      {
        for (unsigned k = 1; k <= 5; ++k)
        {
          month m(i);
          weekday_indexed wdi = weekday{j}[k];
          month_weekday mwd1  = m / wdi;
          month_weekday mwd2  = wdi / m;
          assert(mwd1.month() == m);
          assert(mwd1.weekday_indexed() == wdi);
          assert(mwd2.month() == m);
          assert(mwd2.weekday_indexed() == wdi);
          assert(mwd1 == mwd2);
        }
      }
    }
  }

  { // operator/(int m, const weekday_indexed& wdi) (and switched)
    static_assert(noexcept(2 / Tuesday[2]));
    static_assert(cuda::std::is_same_v<month_weekday, decltype(2 / Tuesday[2])>);
    static_assert(noexcept(Tuesday[2] / 2));
    static_assert(cuda::std::is_same_v<month_weekday, decltype(Tuesday[2] / 2)>);

    //  Run the example
    {
      constexpr month_weekday wdi = 2 / Tuesday[3];
      static_assert(wdi.month() == February, "");
      static_assert(wdi.weekday_indexed() == Tuesday[3], "");
    }

    for (int i = 1; i <= 12; ++i)
    {
      for (unsigned j = 0; j <= 6; ++j)
      {
        for (unsigned k = 1; k <= 5; ++k)
        {
          weekday_indexed wdi = weekday{j}[k];
          month_weekday mwd1  = i / wdi;
          month_weekday mwd2  = wdi / i;
          assert(mwd1.month() == month(i));
          assert(mwd1.weekday_indexed() == wdi);
          assert(mwd2.month() == month(i));
          assert(mwd2.weekday_indexed() == wdi);
          assert(mwd1 == mwd2);
        }
      }
    }
  }

  return 0;
}
