/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
// class month_weekday_last;

// constexpr bool ok() const noexcept;
//  Returns: m_.ok() && wdl_.ok().

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month              = cuda::std::chrono::month;
  using weekday            = cuda::std::chrono::weekday;
  using weekday_last       = cuda::std::chrono::weekday_last;
  using month_weekday_last = cuda::std::chrono::month_weekday_last;

  constexpr month January            = cuda::std::chrono::January;
  constexpr weekday Tuesday          = cuda::std::chrono::Tuesday;
  constexpr weekday_last lastTuesday = weekday_last{Tuesday};

  static_assert(noexcept(cuda::std::declval<const month_weekday_last>().ok()));
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::declval<const month_weekday_last>().ok())>);

  static_assert(!month_weekday_last{month{}, lastTuesday}.ok(), ""); // Bad month
  static_assert(!month_weekday_last{January, weekday_last{weekday{12}}}.ok(), ""); // Bad month
  static_assert(month_weekday_last{January, lastTuesday}.ok(), ""); // Both OK

  for (unsigned i = 0; i <= 50; ++i)
  {
    month_weekday_last mwdl{month{i}, lastTuesday};
    assert(mwdl.ok() == month{i}.ok());
  }

  for (unsigned i = 0; i <= 50; ++i)
  {
    month_weekday_last mwdl{January, weekday_last{weekday{i}}};
    assert(mwdl.ok() == weekday_last{weekday{i}}.ok());
  }

  return 0;
}
