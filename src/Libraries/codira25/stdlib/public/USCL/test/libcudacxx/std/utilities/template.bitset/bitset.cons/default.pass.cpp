/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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

// default ctor

#include <uscl/std/bitset>
#include <uscl/std/cassert>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero
template <cuda::std::size_t N>
__host__ __device__ constexpr void test_default_ctor()
{
  {
    constexpr cuda::std::bitset<N> v1;
    assert(v1.size() == N);
    for (cuda::std::size_t i = 0; i < v1.size(); ++i)
    {
      {
        assert(v1[i] == false);
      }
    }
  }
#if TEST_STD_VER >= 11
  {
    constexpr cuda::std::bitset<N> v1;
    static_assert(v1.size() == N, "");
  }
#endif
}

__host__ __device__ constexpr bool test()
{
  test_default_ctor<0>();
  test_default_ctor<1>();
  test_default_ctor<31>();
  test_default_ctor<32>();
  test_default_ctor<33>();
  test_default_ctor<63>();
  test_default_ctor<64>();
  test_default_ctor<65>();
  test_default_ctor<1000>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
