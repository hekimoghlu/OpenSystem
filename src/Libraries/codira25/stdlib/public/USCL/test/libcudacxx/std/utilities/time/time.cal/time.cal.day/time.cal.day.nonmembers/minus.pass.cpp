/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

// constexpr day operator-(const day& x, const days& y) noexcept;
//   Returns: x + -y.
//
// constexpr days operator-(const day& x, const day& y) noexcept;
//   Returns: days{int(unsigned{x}) - int(unsigned{y}).

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4307) // potential overflow
TEST_DIAG_SUPPRESS_MSVC(4308) // unsigned/signed comparisons

template <typename D, typename Ds>
__host__ __device__ constexpr bool testConstexpr()
{
  D d{23};
  Ds offset{6};
  if (d - offset != D{17})
  {
    return false;
  }
  if (d - D{17} != offset)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using day  = cuda::std::chrono::day;
  using days = cuda::std::chrono::days;

  static_assert(noexcept(cuda::std::declval<day>() - cuda::std::declval<days>()));
  static_assert(noexcept(cuda::std::declval<day>() - cuda::std::declval<day>()));

  static_assert(cuda::std::is_same_v<day, decltype(cuda::std::declval<day>() - cuda::std::declval<days>())>);
  static_assert(cuda::std::is_same_v<days, decltype(cuda::std::declval<day>() - cuda::std::declval<day>())>);

  static_assert(testConstexpr<day, days>(), "");

  day dy{12};
  for (unsigned i = 0; i <= 10; ++i)
  {
    day d1   = dy - days{i};
    days off = dy - day{i};
    assert(static_cast<unsigned>(d1) == 12 - i);
    assert(off.count() == static_cast<int>(12 - i)); // days is signed
  }

  return 0;
}
