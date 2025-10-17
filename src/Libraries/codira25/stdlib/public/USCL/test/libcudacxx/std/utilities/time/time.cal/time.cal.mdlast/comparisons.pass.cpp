/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
// class month_day_last;

// constexpr bool operator==(const month_day& x, const month_day& y) noexcept;
//   Returns: x.month() == y.month()
//
// constexpr bool operator< (const month_day& x, const month_day& y) noexcept;
//   Returns: x.month() < y.month()

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month          = cuda::std::chrono::month;
  using month_day_last = cuda::std::chrono::month_day_last;

  AssertComparisonsAreNoexcept<month_day_last>();
  AssertComparisonsReturnBool<month_day_last>();

  static_assert(testComparisonsValues<month_day_last>(month{1}, month{1}), "");
  static_assert(testComparisonsValues<month_day_last>(month{1}, month{2}), "");

  //  same day, different months
  for (unsigned i = 1; i < 12; ++i)
  {
    for (unsigned j = 1; j < 12; ++j)
    {
      assert((testComparisonsValues<month_day_last>(month{i}, month{j})));
    }
  }

  return 0;
}
