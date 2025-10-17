/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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

// constexpr chrono::weekday_indexed weekday_indexed() const noexcept;
//  Returns: wdi_

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month_weekday   = cuda::std::chrono::month_weekday;
  using month           = cuda::std::chrono::month;
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  constexpr weekday Sunday = cuda::std::chrono::Sunday;

  static_assert(noexcept(cuda::std::declval<const month_weekday>().weekday_indexed()));
  static_assert(
    cuda::std::is_same_v<weekday_indexed, decltype(cuda::std::declval<const month_weekday>().weekday_indexed())>);

  static_assert(month_weekday{month{}, weekday_indexed{}}.weekday_indexed() == weekday_indexed{}, "");

  for (unsigned i = 1; i <= 10; ++i)
  {
    constexpr month March = cuda::std::chrono::March;
    month_weekday md(March, weekday_indexed{Sunday, i});
    assert(static_cast<unsigned>(md.weekday_indexed().weekday() == Sunday));
    assert(static_cast<unsigned>(md.weekday_indexed().index() == i));
  }

  return 0;
}
