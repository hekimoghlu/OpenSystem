/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
// class weekday;

//                     weekday() = default;
//  explicit constexpr weekday(unsigned wd) noexcept;
//  constexpr weekday(const sys_days& dp) noexcept;
//  explicit constexpr weekday(const local_days& dp) noexcept;
//
//  unsigned c_encoding() const noexcept;

//  Effects: Constructs an object of type weekday by initializing wd_ with wd == 7 ? 0 : wd
//    The value held is unspecified if wd is not in the range [0, 255].

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using weekday = cuda::std::chrono::weekday;

  static_assert(noexcept(weekday{}));
  static_assert(noexcept(weekday(1)));
  static_assert(noexcept(weekday(1).c_encoding()));

  constexpr weekday m0{};
  static_assert(m0.c_encoding() == 0, "");

  constexpr weekday m1{1};
  static_assert(m1.c_encoding() == 1, "");

  for (unsigned i = 0; i <= 255; ++i)
  {
    weekday m(i);
    assert(m.c_encoding() == (i == 7 ? 0 : i));
  }

  // TODO - sys_days and local_days ctor tests

  return 0;
}
