/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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
// class weekday_indexed;

// constexpr bool operator==(const weekday_indexed& x, const weekday_indexed& y) noexcept;
//   Returns: x.weekday() == y.weekday() && x.index() == y.index().
// constexpr bool operator!=(const weekday_indexed& x, const weekday_indexed& y) noexcept;
//   Returns: !(x == y)

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  AssertEqualityAreNoexcept<weekday_indexed>();
  AssertEqualityReturnBool<weekday_indexed>();

  static_assert((weekday_indexed{} == weekday_indexed{}), "");
  static_assert(!(weekday_indexed{} != weekday_indexed{}), "");

  static_assert(!(weekday_indexed{} == weekday_indexed{cuda::std::chrono::Tuesday, 1}), "");
  static_assert((weekday_indexed{} != weekday_indexed{cuda::std::chrono::Tuesday, 1}), "");

  //  Some 'ok' values as well
  static_assert((weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{1}, 2}), "");
  static_assert(!(weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{1}, 2}), "");

  static_assert(!(weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{1}, 1}), "");
  static_assert((weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{1}, 1}), "");
  static_assert(!(weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{2}, 2}), "");
  static_assert((weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{2}, 2}), "");

  return 0;
}
