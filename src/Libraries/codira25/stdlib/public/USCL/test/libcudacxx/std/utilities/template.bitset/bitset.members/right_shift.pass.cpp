/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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

// bitset<N> operator>>(size_t pos) const; // constexpr since C++23

#include <uscl/std/bitset>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N, cuda::std::size_t Start = 0, cuda::std::size_t End = static_cast<cuda::std::size_t>(-1)>
__host__ __device__ constexpr bool test_right_shift()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  if (Start >= 9)
  {
    assert(End >= cases.size());
  }
  for (cuda::std::size_t c = Start; c != cases.size() && c != End; ++c)
  {
    for (cuda::std::size_t s = 0; s <= N + 1; ++s)
    {
      cuda::std::bitset<N> v1(cases[c]);
      cuda::std::bitset<N> v2 = v1;
      assert((v1 >>= s) == (v2 >> s));
    }
  }
  return true;
}

__host__ __device__ int main(int, char**)
{
  test_right_shift<0>();
  test_right_shift<1>();
  test_right_shift<31>();
  test_right_shift<32>();
  test_right_shift<33>();
  test_right_shift<63>();
  test_right_shift<64>();
  test_right_shift<65>();
  test_right_shift<1000>(); // not in constexpr because of constexpr evaluation step limits
  static_assert(test_right_shift<0>(), "");
  static_assert(test_right_shift<1>(), "");
  static_assert(test_right_shift<31>(), "");
  static_assert(test_right_shift<32>(), "");
  static_assert(test_right_shift<33>(), "");
  static_assert(test_right_shift<63, 0, 6>(), "");
  static_assert(test_right_shift<63, 6>(), "");
  static_assert(test_right_shift<64, 0, 6>(), "");
  static_assert(test_right_shift<64, 6>(), "");
  static_assert(test_right_shift<65, 0, 3>(), "");
  static_assert(test_right_shift<65, 3, 6>(), "");
  static_assert(test_right_shift<65, 6, 9>(), "");
  static_assert(test_right_shift<65, 9>(), "");

  return 0;
}
