/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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

// XFAIL: gcc-4.8, gcc-5, gcc-6
// gcc before gcc-7 fails with an internal compiler error

// <chrono>
// class year_month_weekday;

// constexpr year_month_weekday& operator+=(const months& m) noexcept;
// constexpr year_month_weekday& operator-=(const months& m) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <typename D, typename Ds>
__host__ __device__ constexpr bool testConstexpr(D d1)
{
  if (static_cast<unsigned>((d1).month()) != 1)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{1}).month()) != 2)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{2}).month()) != 4)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{12}).month()) != 4)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{1}).month()) != 3)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{2}).month()) != 1)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{12}).month()) != 1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year               = cuda::std::chrono::year;
  using month              = cuda::std::chrono::month;
  using weekday_indexed    = cuda::std::chrono::weekday_indexed;
  using year_month_weekday = cuda::std::chrono::year_month_weekday;
  using months             = cuda::std::chrono::months;

  static_assert(noexcept(cuda::std::declval<year_month_weekday&>() += cuda::std::declval<months>()));
  static_assert(
    cuda::std::is_same_v<year_month_weekday&,
                         decltype(cuda::std::declval<year_month_weekday&>() += cuda::std::declval<months>())>);

  static_assert(noexcept(cuda::std::declval<year_month_weekday&>() -= cuda::std::declval<months>()));
  static_assert(
    cuda::std::is_same_v<year_month_weekday&,
                         decltype(cuda::std::declval<year_month_weekday&>() -= cuda::std::declval<months>())>);

  auto constexpr Tuesday = cuda::std::chrono::Tuesday;
  static_assert(
    testConstexpr<year_month_weekday, months>(year_month_weekday{year{1234}, month{1}, weekday_indexed{Tuesday, 2}}),
    "");

  for (unsigned i = 0; i <= 10; ++i)
  {
    year y{1234};
    year_month_weekday ymwd(y, month{i}, weekday_indexed{Tuesday, 2});

    assert(static_cast<unsigned>((ymwd += months{2}).month()) == i + 2);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<unsigned>((ymwd).month()) == i + 2);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<unsigned>((ymwd -= months{1}).month()) == i + 1);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<unsigned>((ymwd).month()) == i + 1);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);
  }

  return 0;
}
