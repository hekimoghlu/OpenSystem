/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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

//            year_month() = default;
//  constexpr year_month(const chrono::year& y, const chrono::month& m) noexcept;
//
//  Effects:  Constructs an object of type year_month by initializing y_ with y, and m_ with m.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year       = cuda::std::chrono::year;
  using month      = cuda::std::chrono::month;
  using year_month = cuda::std::chrono::year_month;

  static_assert(noexcept(year_month{}));
  static_assert(noexcept(year_month{year{1}, month{1}}));

  constexpr year_month ym0{};
  static_assert(ym0.year() == year{}, "");
  static_assert(ym0.month() == month{}, "");
  static_assert(!ym0.ok(), "");

  constexpr year_month ym1{year{2018}, cuda::std::chrono::January};
  static_assert(ym1.year() == year{2018}, "");
  static_assert(ym1.month() == cuda::std::chrono::January, "");
  static_assert(ym1.ok(), "");

  constexpr year_month ym2{year{2018}, month{}};
  static_assert(ym2.year() == year{2018}, "");
  static_assert(ym2.month() == month{}, "");
  static_assert(!ym2.ok(), "");

  return 0;
}
