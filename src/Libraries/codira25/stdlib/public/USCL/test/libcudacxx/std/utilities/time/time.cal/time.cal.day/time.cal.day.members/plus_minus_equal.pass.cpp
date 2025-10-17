/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 20, 2022.
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
// class day;

// constexpr day& operator+=(const days& d) noexcept;
// constexpr day& operator-=(const days& d) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4307) // potential overflow
TEST_DIAG_SUPPRESS_MSVC(4308) // unsigned/signed comparisons

template <typename D, typename Ds>
__host__ __device__ constexpr bool testConstexpr()
{
  D d1{1};
  if (static_cast<unsigned>(d1 += Ds{1}) != 2)
  {
    return false;
  }
  if (static_cast<unsigned>(d1 += Ds{2}) != 4)
  {
    return false;
  }
  if (static_cast<unsigned>(d1 += Ds{22}) != 26)
  {
    return false;
  }
  if (static_cast<unsigned>(d1 -= Ds{1}) != 25)
  {
    return false;
  }
  if (static_cast<unsigned>(d1 -= Ds{2}) != 23)
  {
    return false;
  }
  if (static_cast<unsigned>(d1 -= Ds{22}) != 1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using day  = cuda::std::chrono::day;
  using days = cuda::std::chrono::days;

  static_assert(noexcept(cuda::std::declval<day&>() += cuda::std::declval<days>()));
  static_assert(noexcept(cuda::std::declval<day&>() -= cuda::std::declval<days>()));

  static_assert(cuda::std::is_same_v<day&, decltype(cuda::std::declval<day&>() += cuda::std::declval<days>())>);
  static_assert(cuda::std::is_same_v<day&, decltype(cuda::std::declval<day&>() -= cuda::std::declval<days>())>);

  static_assert(testConstexpr<day, days>(), "");

  for (unsigned i = 0; i <= 10; ++i)
  {
    day day(i);
    assert(static_cast<unsigned>(day += days{22}) == i + 22);
    assert(static_cast<unsigned>(day) == i + 22);
    assert(static_cast<unsigned>(day -= days{12}) == i + 10);
    assert(static_cast<unsigned>(day) == i + 10);
  }

  return 0;
}
