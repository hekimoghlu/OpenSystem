/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// move_sentinel

// template<class S2>
//   requires assignable_from<S&, const S2&>
//     constexpr move_sentinel& operator=(const move_sentinel<S2>& s);

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/iterator>

#include "test_macros.h"

struct NonAssignable
{
  __host__ __device__ NonAssignable& operator=(int i);
};
static_assert(cuda::std::semiregular<NonAssignable>);
static_assert(cuda::std::is_assignable_v<NonAssignable, int>);
static_assert(!cuda::std::assignable_from<NonAssignable, int>);

__host__ __device__ constexpr bool test()
{
  // Assigning from an lvalue.
  {
    cuda::std::move_sentinel<int> m(42);
    cuda::std::move_sentinel<long> m2;
    m2 = m;
    assert(m2.base() == 42L);
  }

  // Assigning from an rvalue.
  {
    cuda::std::move_sentinel<long> m2;
    m2 = cuda::std::move_sentinel<int>(43);
    assert(m2.base() == 43L);
  }

  // SFINAE checks.
  {
    static_assert(cuda::std::is_assignable_v<cuda::std::move_sentinel<int>, cuda::std::move_sentinel<long>>);
    static_assert(!cuda::std::is_assignable_v<cuda::std::move_sentinel<int*>, cuda::std::move_sentinel<const int*>>);
    static_assert(cuda::std::is_assignable_v<cuda::std::move_sentinel<const int*>, cuda::std::move_sentinel<int*>>);
    static_assert(!cuda::std::is_assignable_v<cuda::std::move_sentinel<NonAssignable>, cuda::std::move_sentinel<int>>);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
