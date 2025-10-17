/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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

// constexpr bool operator==(const year& x, const year& y) noexcept;
// constexpr bool operator!=(const year& x, const year& y) noexcept;
// constexpr bool operator< (const year& x, const year& y) noexcept;
// constexpr bool operator> (const year& x, const year& y) noexcept;
// constexpr bool operator<=(const year& x, const year& y) noexcept;
// constexpr bool operator>=(const year& x, const year& y) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using year = cuda::std::chrono::year;

  AssertComparisonsAreNoexcept<year>();
  AssertComparisonsReturnBool<year>();

  static_assert(testComparisonsValues<year>(0, 0), "");
  static_assert(testComparisonsValues<year>(0, 1), "");

  //  Some 'ok' values as well
  static_assert(testComparisonsValues<year>(5, 5), "");
  static_assert(testComparisonsValues<year>(5, 10), "");

  for (int i = 1; i < 10; ++i)
  {
    for (int j = 1; j < 10; ++j)
    {
      assert(testComparisonsValues<year>(i, j));
    }
  }

  return 0;
}
