/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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

// constexpr year& operator+=(const years& d) noexcept;
// constexpr year& operator-=(const years& d) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <typename Y, typename Ys>
__host__ __device__ constexpr bool testConstexpr()
{
  Y y1{1};
  if (static_cast<int>(y1 += Ys{1}) != 2)
  {
    return false;
  }
  if (static_cast<int>(y1 += Ys{2}) != 4)
  {
    return false;
  }
  if (static_cast<int>(y1 += Ys{8}) != 12)
  {
    return false;
  }
  if (static_cast<int>(y1 -= Ys{1}) != 11)
  {
    return false;
  }
  if (static_cast<int>(y1 -= Ys{2}) != 9)
  {
    return false;
  }
  if (static_cast<int>(y1 -= Ys{8}) != 1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year  = cuda::std::chrono::year;
  using years = cuda::std::chrono::years;

  static_assert(noexcept(cuda::std::declval<year&>() += cuda::std::declval<years>()));
  static_assert(noexcept(cuda::std::declval<year&>() -= cuda::std::declval<years>()));

  static_assert(cuda::std::is_same_v<year&, decltype(cuda::std::declval<year&>() += cuda::std::declval<years>())>);
  static_assert(cuda::std::is_same_v<year&, decltype(cuda::std::declval<year&>() -= cuda::std::declval<years>())>);

  static_assert(testConstexpr<year, years>(), "");

  for (int i = 10000; i <= 10020; ++i)
  {
    year year(i);
    assert(static_cast<int>(year += years{10}) == i + 10);
    assert(static_cast<int>(year) == i + 10);
    assert(static_cast<int>(year -= years{9}) == i + 1);
    assert(static_cast<int>(year) == i + 1);
  }

  return 0;
}
