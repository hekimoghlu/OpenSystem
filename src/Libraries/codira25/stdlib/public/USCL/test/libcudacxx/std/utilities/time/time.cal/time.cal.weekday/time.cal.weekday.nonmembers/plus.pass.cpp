/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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

// constexpr weekday operator+(const days& x, const weekday& y) noexcept;
//   Returns: weekday(int{x} + y.count()).
//
// constexpr weekday operator+(const weekday& x, const days& y) noexcept;
//   Returns:
//      weekday{modulo(static_cast<long long>(unsigned{x}) + y.count(), 7)}
//   where modulo(n, 7) computes the remainder of n divided by 7 using Euclidean division.
//   [Note: Given a divisor of 12, Euclidean division truncates towards negative infinity
//   and always produces a remainder in the range of [0, 6].
//   Assuming no overflow in the signed summation, this operation results in a weekday
//   holding a value in the range [0, 6] even if !x.ok(). â€”end note]
//   [Example: Monday + days{6} == Sunday. â€”end example]

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "../../euclidian.h"
#include "test_macros.h"

template <typename M, typename Ms>
__host__ __device__ constexpr bool testConstexpr()
{
  M m{1};
  Ms offset{4};
  if (m + offset != M{5})
  {
    return false;
  }
  if (offset + m != M{5})
  {
    return false;
  }
  //  Check the example
  if (M{1} + Ms{6} != M{0})
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using weekday = cuda::std::chrono::weekday;
  using days    = cuda::std::chrono::days;

  static_assert(noexcept(cuda::std::declval<weekday>() + cuda::std::declval<days>()));
  static_assert(cuda::std::is_same_v<weekday, decltype(cuda::std::declval<weekday>() + cuda::std::declval<days>())>);

  static_assert(noexcept(cuda::std::declval<days>() + cuda::std::declval<weekday>()));
  static_assert(cuda::std::is_same_v<weekday, decltype(cuda::std::declval<days>() + cuda::std::declval<weekday>())>);

  static_assert(testConstexpr<weekday, days>(), "");

  for (unsigned i = 0; i <= 6; ++i)
  {
    for (unsigned j = 0; j <= 6; ++j)
    {
      weekday wd1 = weekday{i} + days{j};
      weekday wd2 = days{j} + weekday{i};
      assert(wd1 == wd2);
      assert((wd1.c_encoding() == euclidian_addition<unsigned, 0, 6>(i, j)));
      assert((wd2.c_encoding() == euclidian_addition<unsigned, 0, 6>(i, j)));
    }
  }

  return 0;
}
