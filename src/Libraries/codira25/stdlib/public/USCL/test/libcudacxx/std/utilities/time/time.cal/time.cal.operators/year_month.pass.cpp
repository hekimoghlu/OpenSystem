/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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
// class year_month;

// constexpr year_month operator/(const year& y, const month& m) noexcept;
//   Returns: {y, m}.
//
// constexpr year_month operator/(const year& y, int m) noexcept;
//   Returns: y / month(m).

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month      = cuda::std::chrono::month;
  using year       = cuda::std::chrono::year;
  using year_month = cuda::std::chrono::year_month;

  constexpr month February = cuda::std::chrono::February;

  { // operator/(const year& y, const month& m)
    static_assert(noexcept(year{2018} / February));
    static_assert(cuda::std::is_same_v<year_month, decltype(year{2018} / February)>);

    static_assert((year{2018} / February).year() == year{2018}, "");
    static_assert((year{2018} / February).month() == month{2}, "");
    for (int i = 1000; i <= 1030; ++i)
    {
      for (unsigned j = 1; j <= 12; ++j)
      {
        year_month ym = year{i} / month{j};
        assert(static_cast<int>(ym.year()) == i);
        assert(static_cast<unsigned>(ym.month()) == j);
      }
    }
  }

  { // operator/(const year& y, const int m)
    static_assert(noexcept(year{2018} / 4));
    static_assert(cuda::std::is_same_v<year_month, decltype(year{2018} / 4)>);

    static_assert((year{2018} / 2).year() == year{2018}, "");
    static_assert((year{2018} / 2).month() == month{2}, "");

    for (int i = 1000; i <= 1030; ++i)
    {
      for (unsigned j = 1; j <= 12; ++j)
      {
        year_month ym = year{i} / j;
        assert(static_cast<int>(ym.year()) == i);
        assert(static_cast<unsigned>(ym.month()) == j);
      }
    }
  }

  return 0;
}
