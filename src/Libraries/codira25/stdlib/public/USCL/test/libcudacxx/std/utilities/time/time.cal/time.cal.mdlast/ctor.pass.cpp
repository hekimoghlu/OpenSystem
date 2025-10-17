/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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

//  constexpr month_day_last(const chrono::month& m) noexcept;
//
//  Effects:  Constructs an object of type month_day_last by initializing m_ with m
//
//  constexpr chrono::month month() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month          = cuda::std::chrono::month;
  using month_day_last = cuda::std::chrono::month_day_last;

  static_assert(noexcept(month_day_last{month{1}}));

  constexpr month_day_last md0{month{}};
  static_assert(md0.month() == month{}, "");
  static_assert(!md0.ok(), "");

  constexpr month_day_last md1{cuda::std::chrono::January};
  static_assert(md1.month() == cuda::std::chrono::January, "");
  static_assert(md1.ok(), "");

  return 0;
}
