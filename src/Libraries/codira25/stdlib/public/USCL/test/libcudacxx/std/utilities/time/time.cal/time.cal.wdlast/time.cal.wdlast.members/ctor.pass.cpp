/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
// class weekday_last;

//  explicit constexpr weekday_last(const chrono::weekday& wd) noexcept;
//
//  Effects: Constructs an object of type weekday_last by initializing wd_ with wd.
//
//  constexpr chrono::weekday weekday() const noexcept;
//  constexpr bool ok() const noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using weekday      = cuda::std::chrono::weekday;
  using weekday_last = cuda::std::chrono::weekday_last;

  static_assert(noexcept(weekday_last{weekday{}}));

  constexpr weekday_last wdl0{weekday{}};
  static_assert(wdl0.weekday() == weekday{}, "");
  static_assert(wdl0.ok(), "");

  constexpr weekday_last wdl1{weekday{1}};
  static_assert(wdl1.weekday() == weekday{1}, "");
  static_assert(wdl1.ok(), "");

  for (unsigned i = 0; i <= 255; ++i)
  {
    weekday_last wdl{weekday{i}};
    assert(wdl.weekday() == weekday{i});
  }

  return 0;
}
