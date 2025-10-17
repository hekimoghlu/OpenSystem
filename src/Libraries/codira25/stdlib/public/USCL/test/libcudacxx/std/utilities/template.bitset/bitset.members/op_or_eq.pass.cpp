/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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

// CONSTEXPR_STEPS: 15000000

// bitset<N>& operator|=(const bitset<N>& rhs); // constexpr since C++23

#include <uscl/std/bitset>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N, cuda::std::size_t Start = 0, cuda::std::size_t End = static_cast<cuda::std::size_t>(-1)>
__host__ __device__ constexpr bool test_op_or_eq()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  if (Start != 0)
  {
    assert(End >= cases.size());
  }
  for (cuda::std::size_t c1 = Start; c1 != cases.size() && c1 != End; ++c1)
  {
    for (cuda::std::size_t c2 = 0; c2 != cases.size(); ++c2)
    {
      cuda::std::bitset<N> v1(cases[c1]);
      cuda::std::bitset<N> v2(cases[c2]);
      cuda::std::bitset<N> v3 = v1;
      v1 |= v2;
      for (cuda::std::size_t i = 0; i < v1.size(); ++i)
      {
        {
          assert(v1[i] == (v3[i] || v2[i]));
        }
      }
    }
  }

  return true;
}

int main(int, char**)
{
  test_op_or_eq<0>();
  test_op_or_eq<1>();
  test_op_or_eq<31>();
  test_op_or_eq<32>();
  test_op_or_eq<33>();
  test_op_or_eq<63>();
  test_op_or_eq<64>();
  test_op_or_eq<65>();
  test_op_or_eq<1000>(); // not in constexpr because of constexpr evaluation step limits
  static_assert(test_op_or_eq<0>(), "");
  static_assert(test_op_or_eq<1>(), "");
  static_assert(test_op_or_eq<31>(), "");
  static_assert(test_op_or_eq<32>(), "");
  static_assert(test_op_or_eq<33>(), "");
  static_assert(test_op_or_eq<63>(), "");
  static_assert(test_op_or_eq<64>(), "");
  static_assert(test_op_or_eq<65, 0, 6>(), "");
  static_assert(test_op_or_eq<65, 6>(), "");

  return 0;
}
