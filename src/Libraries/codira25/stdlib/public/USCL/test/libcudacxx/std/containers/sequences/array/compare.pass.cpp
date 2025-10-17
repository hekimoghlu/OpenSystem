/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// bool operator==(array<T, N> const&, array<T, N> const&);   // constexpr in C++20
// bool operator!=(array<T, N> const&, array<T, N> const&);   // constexpr in C++20
// bool operator<(array<T, N> const&, array<T, N> const&);    // constexpr in C++20
// bool operator<=(array<T, N> const&, array<T, N> const&);   // constexpr in C++20
// bool operator>(array<T, N> const&, array<T, N> const&);    // constexpr in C++20
// bool operator>=(array<T, N> const&, array<T, N> const&);   // constexpr in C++20

#include <uscl/std/array>
#include <uscl/std/cassert>

#include "test_comparisons.h"
#include "test_macros.h"

__host__ __device__ constexpr bool tests()
{
  {
    typedef cuda::std::array<int, 3> C;
    const C c1 = {1, 2, 3};
    const C c2 = {1, 2, 3};
    const C c3 = {3, 2, 1};
    const C c4 = {1, 2, 1};
    assert(testComparisons(c1, c2, true, false));
    assert(testComparisons(c1, c3, false, true));
    assert(testComparisons(c1, c4, false, false));
  }
  {
    typedef cuda::std::array<int, 0> C;
    const C c1 = {};
    const C c2 = {};
    assert(testComparisons(c1, c2, true, false));
  }
  {
    typedef cuda::std::array<LessAndEqComp, 3> C;
    const C c1 = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(3)};
    const C c2 = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(3)};
    const C c3 = {LessAndEqComp(3), LessAndEqComp(2), LessAndEqComp(1)};
    const C c4 = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(1)};
    assert(testComparisons(c1, c2, true, false));
    assert(testComparisons(c1, c3, false, true));
    assert(testComparisons(c1, c4, false, false));
  }
  {
    typedef cuda::std::array<LessAndEqComp, 0> C;
    const C c1 = {};
    const C c2 = {};
    assert(testComparisons(c1, c2, true, false));
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
