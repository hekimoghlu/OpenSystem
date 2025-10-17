/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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

// void swap(array& a);
// namespace std { void swap(array<T, N> &x, array<T, N> &y);

#include <uscl/std/array>
#include <uscl/std/cassert>

#include "test_macros.h"

struct NonSwappable
{
  __host__ __device__ constexpr NonSwappable() {}

private:
  __host__ __device__ NonSwappable(NonSwappable const&);
  __host__ __device__ NonSwappable& operator=(NonSwappable const&);
};

__host__ __device__ constexpr bool tests()
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C c1 = {1, 2, 3.5};
    C c2 = {4, 5, 6.5};
    c1.swap(c2);
    assert(c1.size() == 3);
    assert(c1[0] == 4);
    assert(c1[1] == 5);
    assert(c1[2] == 6.5);
    assert(c2.size() == 3);
    assert(c2[0] == 1);
    assert(c2[1] == 2);
    assert(c2[2] == 3.5);
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C c1 = {1, 2, 3.5};
    C c2 = {4, 5, 6.5};
    cuda::std::swap(c1, c2);
    assert(c1.size() == 3);
    assert(c1[0] == 4);
    assert(c1[1] == 5);
    assert(c1[2] == 6.5);
    assert(c2.size() == 3);
    assert(c2[0] == 1);
    assert(c2[1] == 2);
    assert(c2[2] == 3.5);
  }

  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    C c1 = {};
    C c2 = {};
    c1.swap(c2);
    assert(c1.size() == 0);
    assert(c2.size() == 0);
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    C c1 = {};
    C c2 = {};
    cuda::std::swap(c1, c2);
    assert(c1.size() == 0);
    assert(c2.size() == 0);
  }
  {
    typedef NonSwappable T;
    typedef cuda::std::array<T, 0> C0;
    C0 l = {};
    C0 r = {};
    l.swap(r);
    static_assert(noexcept(l.swap(r)), "");
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
