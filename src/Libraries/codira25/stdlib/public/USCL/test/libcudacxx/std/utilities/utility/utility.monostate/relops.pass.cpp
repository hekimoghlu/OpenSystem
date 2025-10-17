/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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

// <cuda/std/utility>

// constexpr bool operator<(monostate, monostate) noexcept { return false; }
// constexpr bool operator>(monostate, monostate) noexcept { return false; }
// constexpr bool operator<=(monostate, monostate) noexcept { return true; }
// constexpr bool operator>=(monostate, monostate) noexcept { return true; }
// constexpr bool operator==(monostate, monostate) noexcept { return true; }
// constexpr bool operator!=(monostate, monostate) noexcept { return false; }
// constexpr strong_ordering operator<=>(monostate, monostate) noexcept { return strong_ordering::equal; } // since
// C++20

#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "test_comparisons.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using M = cuda::std::monostate;
  constexpr M m1{};
  constexpr M m2{};
  assert(testComparisons(m1, m2, /*isEqual*/ true, /*isLess*/ false));
  AssertComparisonsAreNoexcept<M>();

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  assert(testOrder(m1, m2, cuda::std::strong_ordering::equal));
  AssertOrderAreNoexcept<M>();
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
