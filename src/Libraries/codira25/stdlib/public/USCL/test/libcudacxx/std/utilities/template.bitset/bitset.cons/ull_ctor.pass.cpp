/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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

// bitset(unsigned long long val); // constexpr since C++23

#include <uscl/std/bitset>
#include <uscl/std/cassert>
// #include <uscl/std/algorithm> // for 'min' and 'max'
#include <uscl/std/cstddef>

#include "test_macros.h"

// TEST_MSVC_DIAGNOSTIC_IGNORED(6294) // Ill-defined for-loop:  initial condition does not satisfy test.  Loop body not
// executed.
TEST_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N>
__host__ __device__ constexpr void test_val_ctor()
{
  {
    constexpr cuda::std::bitset<N> v(0xAAAAAAAAAAAAAAAAULL);
    assert(v.size() == N);
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 64);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == ((i & 1) != 0));
    }
    for (cuda::std::size_t i = M; i < v.size(); ++i)
    {
      {
        assert(v[i] == false);
      }
    }
  }
  {
    constexpr cuda::std::bitset<N> v(0xAAAAAAAAAAAAAAAAULL);
    static_assert(v.size() == N, "");
  }
}

__host__ __device__ constexpr bool test()
{
  test_val_ctor<0>();
  test_val_ctor<1>();
  test_val_ctor<31>();
  test_val_ctor<32>();
  test_val_ctor<33>();
  test_val_ctor<63>();
  test_val_ctor<64>();
  test_val_ctor<65>();
  test_val_ctor<1000>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
