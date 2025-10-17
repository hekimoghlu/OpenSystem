/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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

// template<class T>
//   const T&
//   clamp(const T& v, const T& lo, const T& hi);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "test_macros.h"

struct Tag
{
  __host__ __device__ constexpr Tag()
      : val(0)
      , tag("Default")
  {}
  __host__ __device__ constexpr Tag(int a, const char* b)
      : val(a)
      , tag(b)
  {}

  int val;
  const char* tag;
};

__host__ __device__ constexpr bool eq(const Tag& rhs, const Tag& lhs)
{
  return rhs.val == lhs.val && rhs.tag == lhs.tag;
}
__host__ __device__ constexpr bool operator<(const Tag& rhs, const Tag& lhs)
{
  return rhs.val < lhs.val;
}

template <class T>
__host__ __device__ constexpr void test(const T& a, const T& lo, const T& hi, const T& x)
{
  assert(&cuda::std::clamp(a, lo, hi) == &x);
}

__host__ __device__ constexpr bool test()
{
  {
    int x = 0;
    int y = 0;
    int z = 0;
    test(x, y, z, x);
    test(y, x, z, y);
  }
  {
    int x = 0;
    int y = 1;
    int z = 2;
    test(x, y, z, y);
    test(y, x, z, y);
  }
  {
    int x = 1;
    int y = 0;
    int z = 1;
    test(x, y, z, x);
    test(y, x, z, x);
  }

  {
    //  If they're all the same, we should get the value back.
    Tag x{0, "Zero-x"};
    Tag y{0, "Zero-y"};
    Tag z{0, "Zero-z"};
    assert(eq(cuda::std::clamp(x, y, z), x));
    assert(eq(cuda::std::clamp(y, x, z), y));
  }

  {
    //  If it's the same as the lower bound, we get the value back.
    Tag x{0, "Zero-x"};
    Tag y{0, "Zero-y"};
    Tag z{1, "One-z"};
    assert(eq(cuda::std::clamp(x, y, z), x));
    assert(eq(cuda::std::clamp(y, x, z), y));
  }

  {
    //  If it's the same as the upper bound, we get the value back.
    Tag x{1, "One-x"};
    Tag y{0, "Zero-y"};
    Tag z{1, "One-z"};
    assert(eq(cuda::std::clamp(x, y, z), x));
    assert(eq(cuda::std::clamp(z, y, x), z));
  }

  {
    //  If the value is between, we should get the value back
    Tag x{1, "One-x"};
    Tag y{0, "Zero-y"};
    Tag z{2, "Two-z"};
    assert(eq(cuda::std::clamp(x, y, z), x));
    assert(eq(cuda::std::clamp(y, x, z), x));
  }

  {
    //  If the value is less than the 'lo', we should get the lo back.
    Tag x{0, "Zero-x"};
    Tag y{1, "One-y"};
    Tag z{2, "Two-z"};
    assert(eq(cuda::std::clamp(x, y, z), y));
    assert(eq(cuda::std::clamp(y, x, z), y));
  }
  {
    //  If the value is greater than 'hi', we should get hi back.
    Tag x{2, "Two-x"};
    Tag y{0, "Zero-y"};
    Tag z{1, "One-z"};
    assert(eq(cuda::std::clamp(x, y, z), z));
    assert(eq(cuda::std::clamp(y, z, x), z));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
