/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
// class year;

//                     year() = default;
//  explicit constexpr year(int m) noexcept;
//  explicit constexpr operator int() const noexcept;

//  Effects: Constructs an object of type year by initializing y_ with y.
//    The value held is unspecified if d is not in the range [0, 255].

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year = cuda::std::chrono::year;

  static_assert(noexcept(year{}));
  static_assert(noexcept(year(0U)));
  static_assert(noexcept(static_cast<int>(year(0U))));

  constexpr year y0{};
  static_assert(static_cast<int>(y0) == 0, "");

  constexpr year y1{1};
  static_assert(static_cast<int>(y1) == 1, "");

  for (int i = 0; i <= 2550; i += 7)
  {
    year year(i);
    assert(static_cast<int>(year) == i);
  }

  return 0;
}
