/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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

//  constexpr weekday& operator--() noexcept;
//  constexpr weekday operator--(int) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "../../euclidian.h"
#include "test_macros.h"

template <typename WD>
__host__ __device__ constexpr bool testConstexpr()
{
  WD wd{1};
  if ((--wd).c_encoding() != 0)
  {
    return false;
  }
  if ((wd--).c_encoding() != 0)
  {
    return false;
  }
  if ((wd).c_encoding() != 6)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using weekday = cuda::std::chrono::weekday;
  static_assert(noexcept(--(cuda::std::declval<weekday&>())));
  static_assert(noexcept((cuda::std::declval<weekday&>())--));

  static_assert(cuda::std::is_same_v<weekday, decltype(cuda::std::declval<weekday&>()--)>);
  static_assert(cuda::std::is_same_v<weekday&, decltype(--cuda::std::declval<weekday&>())>);

  static_assert(testConstexpr<weekday>(), "");

  for (unsigned i = 0; i <= 6; ++i)
  {
    weekday wd(i);
    assert(((--wd).c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, 1)));
    assert(((wd--).c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, 1)));
    assert(((wd).c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, 2)));
  }

  return 0;
}
