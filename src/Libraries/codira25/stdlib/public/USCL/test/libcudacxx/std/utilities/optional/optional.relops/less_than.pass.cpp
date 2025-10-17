/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// template <class T, class U> constexpr bool operator< (const optional<T>& x, const optional<U>& y);

#include <uscl/std/cassert>
#include <uscl/std/optional>

#include "test_macros.h"

using cuda::std::optional;

struct X
{
  int i_;

  __host__ __device__ constexpr X(int i)
      : i_(i)
  {}
};

__host__ __device__ constexpr bool operator<(const X& lhs, const X& rhs)
{
  return lhs.i_ < rhs.i_;
}

template <class T>
__host__ __device__ constexpr void test()
{
  {
    using O = optional<T>;
    cuda::std::remove_reference_t<T> one{1};
    cuda::std::remove_reference_t<T> two{2};

    O o1{}; // disengaged
    O o2{}; // disengaged
    O o3{one}; // engaged
    O o4{two}; // engaged
    O o5{one}; // engaged

    assert(!(o1 < o1));
    assert(!(o1 < o2));
    assert((o1 < o3));
    assert((o1 < o4));
    assert((o1 < o5));

    assert(!(o2 < o1));
    assert(!(o2 < o2));
    assert((o2 < o3));
    assert((o2 < o4));
    assert((o2 < o5));

    assert(!(o3 < o1));
    assert(!(o3 < o2));
    assert(!(o3 < o3));
    assert((o3 < o4));
    assert(!(o3 < o5));

    assert(!(o4 < o1));
    assert(!(o4 < o2));
    assert(!(o4 < o3));
    assert(!(o4 < o4));
    assert(!(o4 < o5));

    assert(!(o5 < o1));
    assert(!(o5 < o2));
    assert(!(o5 < o3));
    assert((o5 < o4));
    assert(!(o5 < o5));
  }
  {
    using O1 = optional<int>;
    using O2 = optional<long>;
    O1 o1(42);
    assert(o1 < O2(101));
    assert(!(O2(101) < o1));
  }
  {
    using O1 = optional<int>;
    using O2 = optional<const int>;
    O1 o1(42);
    assert(o1 < O2(101));
    assert(!(O2(101) < o1));
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
