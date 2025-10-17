/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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

// bitset<N>& set(); // constexpr since C++23

#include <uscl/std/bitset>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N>
__host__ __device__ constexpr void test_set_all()
{
  cuda::std::bitset<N> v;
  v.set();
  for (cuda::std::size_t i = 0; i < v.size(); ++i)
  {
    {
      assert(v[i]);
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test_set_all<0>();
  test_set_all<1>();
  test_set_all<31>();
  test_set_all<32>();
  test_set_all<33>();
  test_set_all<63>();
  test_set_all<64>();
  test_set_all<65>();
  test_set_all<1000>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
