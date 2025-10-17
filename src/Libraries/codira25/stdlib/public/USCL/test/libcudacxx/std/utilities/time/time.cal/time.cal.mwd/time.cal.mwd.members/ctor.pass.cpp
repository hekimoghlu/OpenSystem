/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
//   month_weekday represents the nth weekday of a month, of an as yet unspecified year.

//  constexpr month_weekday(const chrono::month& m, const chrono::weekday_indexed& wdi) noexcept;
//    Effects:  Constructs an object of type month_weekday by initializing m_ with m, and wdi_ with wdi.
//
//  constexpr chrono::month                     month() const noexcept;
//  constexpr chrono::weekday_indexed weekday_indexed() const noexcept;
//  constexpr bool                                 ok() const noexcept;

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

  static_assert(noexcept(month_weekday{month{1}, weekday_indexed{weekday{}, 1}}));

  constexpr month_weekday md0{month{}, weekday_indexed{}};
  static_assert(md0.month() == month{}, "");
  static_assert(md0.weekday_indexed() == weekday_indexed{}, "");
  static_assert(!md0.ok(), "");

  constexpr month_weekday md1{cuda::std::chrono::January, weekday_indexed{cuda::std::chrono::Friday, 4}};
  static_assert(md1.month() == cuda::std::chrono::January, "");
  static_assert(md1.weekday_indexed() == weekday_indexed{cuda::std::chrono::Friday, 4}, "");
  static_assert(md1.ok(), "");

  return 0;
}
