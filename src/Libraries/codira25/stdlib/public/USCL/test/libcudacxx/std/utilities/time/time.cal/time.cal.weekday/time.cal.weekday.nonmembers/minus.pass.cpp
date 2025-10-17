/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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

// constexpr weekday operator-(const weekday& x, const days& y) noexcept;
//   Returns: x + -y.
//
// constexpr days operator-(const weekday& x, const weekday& y) noexcept;
// Returns: If x.ok() == true and y.ok() == true, returns a value d in the range
//    [days{0}, days{6}] satisfying y + d == x.
// Otherwise the value returned is unspecified.
// [Example: Sunday - Monday == days{6}. â€”end example]

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "../../euclidian.h"
#include "test_macros.h"

template <typename WD, typename Ds>
__host__ __device__ constexpr bool testConstexpr()
{
  {
    WD wd{5};
    Ds offset{3};
    if (wd - offset != WD{2})
    {
      return false;
    }
    if (wd - WD{2} != offset)
    {
      return false;
    }
  }

  //  Check the example
  if (WD{0} - WD{1} != Ds{6})
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using weekday = cuda::std::chrono::weekday;
  using days    = cuda::std::chrono::days;

  static_assert(noexcept(cuda::std::declval<weekday>() - cuda::std::declval<days>()));
  static_assert(cuda::std::is_same_v<weekday, decltype(cuda::std::declval<weekday>() - cuda::std::declval<days>())>);

  static_assert(noexcept(cuda::std::declval<weekday>() - cuda::std::declval<weekday>()));
  static_assert(cuda::std::is_same_v<days, decltype(cuda::std::declval<weekday>() - cuda::std::declval<weekday>())>);

  static_assert(testConstexpr<weekday, days>(), "");

  for (unsigned i = 0; i <= 6; ++i)
  {
    for (unsigned j = 0; j <= 6; ++j)
    {
      weekday wd = weekday{i} - days{j};
      assert(wd + days{j} == weekday{i});
      assert((wd.c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, j)));
    }
  }

  for (unsigned i = 0; i <= 6; ++i)
  {
    for (unsigned j = 0; j <= 6; ++j)
    {
      days d = weekday{j} - weekday{i};
      assert(weekday{i} + d == weekday{j});
    }
  }

  return 0;
}
