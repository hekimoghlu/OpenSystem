/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
// class month;

// constexpr month operator+(const month& x, const months& y) noexcept;
//   Returns: month(int{x} + y.count()).
//
// constexpr month operator+(const months& x, const month& y) noexcept;
//   Returns:
//     month{modulo(static_cast<long long>(int{x}) + (y.count() - 1), 12) + 1}
//   where modulo(n, 12) computes the remainder of n divided by 12 using Euclidean division.
//   [Note: Given a divisor of 12, Euclidean division truncates towards negative infinity
//   and always produces a remainder in the range of [0, 11].
//   Assuming no overflow in the signed summation, this operation results in a month
//   holding a value in the range [1, 12] even if !x.ok(). â€”end note]
//   [Example: February + months{11} == January. â€”end example]

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

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
  if (M{2} + Ms{11} != M{1})
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using month  = cuda::std::chrono::month;
  using months = cuda::std::chrono::months;

  static_assert(noexcept(cuda::std::declval<month>() + cuda::std::declval<months>()));
  static_assert(noexcept(cuda::std::declval<months>() + cuda::std::declval<month>()));

  static_assert(cuda::std::is_same_v<month, decltype(cuda::std::declval<month>() + cuda::std::declval<months>())>);
  static_assert(cuda::std::is_same_v<month, decltype(cuda::std::declval<months>() + cuda::std::declval<month>())>);

  static_assert(testConstexpr<month, months>(), "");

  month my{2};
  for (unsigned i = 0; i <= 15; ++i)
  {
    month m1 = my + months{i};
    month m2 = months{i} + my;
    assert(m1 == m2);
    unsigned exp = i + 2;
    while (exp > 12)
    {
      exp -= 12;
    }
    assert(static_cast<unsigned>(m1) == exp);
    assert(static_cast<unsigned>(m2) == exp);
  }

  return 0;
}
