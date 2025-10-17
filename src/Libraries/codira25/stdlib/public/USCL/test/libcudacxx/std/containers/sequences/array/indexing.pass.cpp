/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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

// reference operator[](size_type); // constexpr in C++17
// Libc++ marks it as noexcept

#include <uscl/std/array>
#include <uscl/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool tests()
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C c = {1, 2, 3.5};
    static_assert(noexcept(c[0]));
    static_assert(cuda::std::is_same_v<C::reference, decltype(c[0])>);
    C::reference r1 = c[0];
    assert(r1 == 1);
    r1 = 5.5;
    assert(c.front() == 5.5);

    C::reference r2 = c[2];
    assert(r2 == 3.5);
    r2 = 7.5;
    assert(c.back() == 7.5);
  }

  // Test operator[] "works" on zero sized arrays
  {
    {
      typedef double T;
      typedef cuda::std::array<T, 0> C;
      C c = {};
      static_assert(noexcept(c[0]));
      static_assert(cuda::std::is_same_v<C::reference, decltype(c[0])>);
      if (c.size() > (0))
      { // always false
#if !TEST_COMPILER(MSVC)
        C::reference r = c[0];
        unused(r);
#endif // !TEST_COMPILER(MSVC)
      }
    }
    {
      typedef double T;
      typedef cuda::std::array<const T, 0> C;
      C c = {};
      static_assert(noexcept(c[0]));
      static_assert(cuda::std::is_same_v<C::reference, decltype(c[0])>);
      if (c.size() > (0))
      { // always false
#if !TEST_COMPILER(MSVC)
        C::reference r = c[0];
        unused(r);
#endif // !TEST_COMPILER(MSVC)
      }
    }
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
