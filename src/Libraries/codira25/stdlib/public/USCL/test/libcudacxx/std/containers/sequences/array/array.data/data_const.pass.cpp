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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// const T* data() const;

#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/cstddef> // for cuda::std::max_align_t
#include <uscl/std/cstdint>

#include "test_macros.h"

struct NoDefault
{
  __host__ __device__ constexpr NoDefault(int) {}
};

__host__ __device__ constexpr bool tests()
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    const C c = {1, 2, 3.5};
    static_assert(noexcept(c.data()));
    const T* p = c.data();
    assert(p[0] == 1);
    assert(p[1] == 2);
    assert(p[2] == 3.5);
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    const C c = {};
    static_assert(noexcept(c.data()));
    const T* p = c.data();
    unused(p);
  }
  {
    typedef NoDefault T;
    typedef cuda::std::array<T, 0> C;
    const C c = {};
    static_assert(noexcept(c.data()));
    const T* p = c.data();
    unused(p);
  }
  {
    cuda::std::array<int, 5> const c = {0, 1, 2, 3, 4};
    assert(c.data() == &c[0]);
    assert(*c.data() == c[0]);
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");

  // Test the alignment of data()
  {
    typedef cuda::std::max_align_t T;
    typedef cuda::std::array<T, 0> C;
    const C c                 = {};
    const T* p                = c.data();
    cuda::std::uintptr_t pint = reinterpret_cast<cuda::std::uintptr_t>(p);
    assert(pint % alignof(T) == 0);
  }

  return 0;
}
