/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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

// constexpr bool ok() const noexcept;
//  Returns: m_.ok() && y_.ok().

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month      = cuda::std::chrono::month;
  using year       = cuda::std::chrono::year;
  using year_month = cuda::std::chrono::year_month;

  constexpr month January = cuda::std::chrono::January;

  static_assert(noexcept(cuda::std::declval<const year_month>().ok()));
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::declval<const year_month>().ok())>);

  static_assert(!year_month{year{-32768}, January}.ok(), ""); // Bad year
  static_assert(!year_month{year{2019}, month{}}.ok(), ""); // Bad month
  static_assert(year_month{year{2019}, January}.ok(), ""); // Both OK

  for (unsigned i = 0; i <= 50; ++i)
  {
    year_month ym{year{2019}, month{i}};
    assert(ym.ok() == month{i}.ok());
  }

  const int ymax = static_cast<int>(year::max());
  for (int i = ymax - 100; i <= ymax + 100; ++i)
  {
    year_month ym{year{i}, January};
    assert(ym.ok() == year{i}.ok());
  }

  return 0;
}
