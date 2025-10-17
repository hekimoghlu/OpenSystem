/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<LessThanComparable T>
//   pair<const T&, const T&>
//   minmax(const T& a, const T& b);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test(const T& a, const T& b, const T& x, const T& y)
{
  cuda::std::pair<const T&, const T&> p = cuda::std::minmax(a, b);
  assert(&p.first == &x);
  assert(&p.second == &y);
}

__host__ __device__ constexpr bool test()
{
  {
    int x = 0;
    int y = 0;
    test(x, y, x, y);
    test(y, x, y, x);
  }
  {
    int x = 0;
    int y = 1;
    test(x, y, x, y);
    test(y, x, x, y);
  }
  {
    int x = 1;
    int y = 0;
    test(x, y, y, x);
    test(y, x, y, x);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
