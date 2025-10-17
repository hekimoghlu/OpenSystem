/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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

// constexpr operator sys_days() const noexcept;
//  Returns: If ok() == true, returns a sys_days that represents the last weekday()
//             of year()/month(). Otherwise the returned value is unspecified.

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                    = cuda::std::chrono::year;
  using year_month_weekday_last = cuda::std::chrono::year_month_weekday_last;
  using sys_days                = cuda::std::chrono::sys_days;
  using days                    = cuda::std::chrono::days;
  using weekday_last            = cuda::std::chrono::weekday_last;

  static_assert(noexcept(static_cast<sys_days>(cuda::std::declval<const year_month_weekday_last>())));
  static_assert(
    cuda::std::is_same_v<sys_days, decltype(static_cast<sys_days>(cuda::std::declval<const year_month_weekday_last>()))>);

  auto constexpr January = cuda::std::chrono::January;
  auto constexpr Tuesday = cuda::std::chrono::Tuesday;

  { // Last Tuesday in Jan 1970 was the 27th
    constexpr year_month_weekday_last ymwdl{year{1970}, January, weekday_last{Tuesday}};
    constexpr sys_days sd{ymwdl};

    static_assert(sd.time_since_epoch() == days{26}, "");
  }

  { // Last Tuesday in Jan 2000 was the 25th
    constexpr year_month_weekday_last ymwdl{year{2000}, January, weekday_last{Tuesday}};
    constexpr sys_days sd{ymwdl};

    static_assert(sd.time_since_epoch() == days{10957 + 24}, "");
  }

  { // Last Tuesday in Jan 1940 was the 30th
    constexpr year_month_weekday_last ymwdl{year{1940}, January, weekday_last{Tuesday}};
    constexpr sys_days sd{ymwdl};

    static_assert(sd.time_since_epoch() == days{-10958 + 29}, "");
  }

  { // Last Tuesday in Nov 1939 was the 28th
    year_month_weekday_last ymdl{year{1939}, cuda::std::chrono::November, weekday_last{Tuesday}};
    sys_days sd{ymdl};

    assert(sd.time_since_epoch() == days{-(10957 + 35)});
  }

  return 0;
}
