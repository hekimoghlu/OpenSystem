/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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

// bool test(size_t pos) const; // constexpr since C++23

#include <uscl/std/bitset>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N>
__host__ __device__ constexpr void test_test()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  for (cuda::std::size_t c = 0; c != cases.size(); ++c)
  {
    cuda::std::bitset<N> const v(cases[c]);
    if (v.size() > 0)
    {
      cuda::std::size_t middle = v.size() / 2;
      bool b                   = v.test(middle);
      assert(b == v[middle]);
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test_test<0>();
  test_test<1>();
  test_test<31>();
  test_test<32>();
  test_test<33>();
  test_test<63>();
  test_test<64>();
  test_test<65>();

  return true;
}

int main(int, char**)
{
  test();
  test_test<1000>(); // not in constexpr because of constexpr evaluation step limits
  static_assert(test(), "");

  return 0;
}
