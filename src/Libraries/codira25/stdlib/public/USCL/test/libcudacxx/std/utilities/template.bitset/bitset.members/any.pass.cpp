/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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

// bool any() const; // constexpr since C++23

#include <uscl/std/bitset>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>

#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__ constexpr void test_any()
{
  cuda::std::bitset<N> v;
  v.reset();
  assert(v.any() == false);
  v.set();
  assert(v.any() == (N != 0));
  if (v.size() > 1)
  {
    v[N / 2] = false;
    assert(v.any() == true);
    v.reset();
    v[N / 2] = true;
    assert(v.any() == true);
  }
}

__host__ __device__ constexpr bool test()
{
  test_any<0>();
  test_any<1>();
  test_any<31>();
  test_any<32>();
  test_any<33>();
  test_any<63>();
  test_any<64>();
  test_any<65>();
  test_any<1000>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
