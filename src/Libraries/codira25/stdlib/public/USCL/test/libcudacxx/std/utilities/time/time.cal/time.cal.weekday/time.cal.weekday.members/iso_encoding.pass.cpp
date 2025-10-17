/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

//  constexpr unsigned iso_encoding() const noexcept;
//  Returns the underlying weekday, _except_ that returns '7' for Sunday (zero)
//    See [time.cal.wd.members]

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <typename WD>
__host__ __device__ constexpr bool testConstexpr()
{
  WD wd{5};
  return wd.c_encoding() == 5;
}

int main(int, char**)
{
  using weekday = cuda::std::chrono::weekday;

  static_assert(noexcept(cuda::std::declval<weekday&>().iso_encoding()));
  static_assert(cuda::std::is_same_v<unsigned, decltype(cuda::std::declval<weekday&>().iso_encoding())>);

  static_assert(testConstexpr<weekday>(), "");

  //  This is different than all the other tests, because the '7' gets converted to
  //  a zero in the constructor, but then back to '7' by iso_encoding().
  for (unsigned i = 0; i <= 10; ++i)
  {
    weekday wd(i);
    assert(wd.iso_encoding() == (i == 0 ? 7 : i));
  }

  return 0;
}
