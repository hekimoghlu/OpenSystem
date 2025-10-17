/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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

// constexpr year operator+(const year& x, const years& y) noexcept;
//   Returns: year(int{x} + y.count()).
//
// constexpr year operator+(const years& x, const year& y) noexcept;
//   Returns: y + x

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <typename Y, typename Ys>
__host__ __device__ constexpr bool testConstexpr()
{
  Y y{1001};
  Ys offset{23};
  if (y + offset != Y{1024})
  {
    return false;
  }
  if (offset + y != Y{1024})
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year  = cuda::std::chrono::year;
  using years = cuda::std::chrono::years;

  static_assert(noexcept(cuda::std::declval<year>() + cuda::std::declval<years>()));
  static_assert(cuda::std::is_same_v<year, decltype(cuda::std::declval<year>() + cuda::std::declval<years>())>);

  static_assert(noexcept(cuda::std::declval<years>() + cuda::std::declval<year>()));
  static_assert(cuda::std::is_same_v<year, decltype(cuda::std::declval<years>() + cuda::std::declval<year>())>);

  static_assert(testConstexpr<year, years>(), "");

  year y{1223};
  for (int i = 1100; i <= 1110; ++i)
  {
    year y1 = y + years{i};
    year y2 = years{i} + y;
    assert(y1 == y2);
    assert(static_cast<int>(y1) == i + 1223);
    assert(static_cast<int>(y2) == i + 1223);
  }

  return 0;
}
