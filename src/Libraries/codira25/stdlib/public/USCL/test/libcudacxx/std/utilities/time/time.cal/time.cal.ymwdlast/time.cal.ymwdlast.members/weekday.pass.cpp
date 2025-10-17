/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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

// constexpr chrono::weekday weekday() const noexcept;
//  Returns: wdi_.weekday()

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

  static_assert(noexcept(cuda::std::declval<const year_month_weekday_last>().weekday()));
  static_assert(cuda::std::is_same_v<weekday, decltype(cuda::std::declval<const year_month_weekday_last>().weekday())>);

  static_assert(year_month_weekday_last{year{}, month{}, weekday_last{weekday{}}}.weekday() == weekday{}, "");

  for (unsigned i = 1; i <= 50; ++i)
  {
    year_month_weekday_last ymwdl(year{1}, month{1}, weekday_last{weekday{i}});
    assert(ymwdl.weekday().c_encoding() == (i == 7 ? 0 : i));
  }

  return 0;
}
