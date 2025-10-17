/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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

//  constexpr month& operator--() noexcept;
//  constexpr month operator--(int) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <typename M>
__host__ __device__ constexpr bool testConstexpr()
{
  M m1{10};
  if (static_cast<unsigned>(--m1) != 9)
  {
    return false;
  }
  if (static_cast<unsigned>(m1--) != 9)
  {
    return false;
  }
  if (static_cast<unsigned>(m1) != 8)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using month = cuda::std::chrono::month;

  static_assert(noexcept(--(cuda::std::declval<month&>())));
  static_assert(noexcept((cuda::std::declval<month&>())--));

  static_assert(cuda::std::is_same_v<month, decltype(cuda::std::declval<month&>()--)>);
  static_assert(cuda::std::is_same_v<month&, decltype(--cuda::std::declval<month&>())>);

  static_assert(testConstexpr<month>(), "");

  for (unsigned i = 10; i <= 20; ++i)
  {
    month month(i);
    assert(static_cast<unsigned>(--month) == i - 1);
    assert(static_cast<unsigned>(month--) == i - 1);
    assert(static_cast<unsigned>(month) == i - 2);
  }

  return 0;
}
