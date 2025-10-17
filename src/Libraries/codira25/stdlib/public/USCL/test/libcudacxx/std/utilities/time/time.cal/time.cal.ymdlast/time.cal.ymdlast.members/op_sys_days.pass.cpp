/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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

// constexpr operator sys_days() const noexcept;
//  Returns: sys_days{year()/month()/day()}.

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                = cuda::std::chrono::year;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;
  using sys_days            = cuda::std::chrono::sys_days;
  using days                = cuda::std::chrono::days;

  static_assert(noexcept(static_cast<sys_days>(cuda::std::declval<const year_month_day_last>())));
  static_assert(
    cuda::std::is_same_v<sys_days, decltype(static_cast<sys_days>(cuda::std::declval<const year_month_day_last>()))>);

  auto constexpr January  = cuda::std::chrono::January;
  auto constexpr November = cuda::std::chrono::November;

  { // Last day in Jan 1970 was the 31st
    constexpr year_month_day_last ymdl{year{1970}, month_day_last{January}};
    constexpr sys_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{30}, "");
  }

  {
    constexpr year_month_day_last ymdl{year{2000}, month_day_last{January}};
    constexpr sys_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{10957 + 30}, "");
  }

  {
    constexpr year_month_day_last ymdl{year{1940}, month_day_last{January}};
    constexpr sys_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{-10957 + 29}, "");
  }

  {
    year_month_day_last ymdl{year{1939}, month_day_last{November}};
    sys_days sd{ymdl};

    assert(sd.time_since_epoch() == days{-(10957 + 33)});
  }

  return 0;
}
