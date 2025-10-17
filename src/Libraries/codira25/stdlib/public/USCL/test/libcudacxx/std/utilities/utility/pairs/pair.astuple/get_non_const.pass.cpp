/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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

// <utility>

// template <class T1, class T2> struct pair

// template<size_t I, class T1, class T2>
//     typename tuple_element<I, cuda::std::pair<T1, T2> >::type&
//     get(pair<T1, T2>&);

// UNSUPPORTED: msvc

#include <uscl/std/cassert>
#include <uscl/std/tuple>
#include <uscl/std/utility>

#include "test_macros.h"

struct S
{
  cuda::std::pair<int, int> a;
  int k;
  __device__ __host__ constexpr S()
      : a{1, 2}
      , k(cuda::std::get<0>(a))
  {}
};

__device__ __host__ constexpr cuda::std::pair<int, int> getP()
{
  return {3, 4};
}

int main(int, char**)
{
  {
    typedef cuda::std::pair<int, short> P;
    P p(3, static_cast<short>(4));
    assert(cuda::std::get<0>(p) == 3);
    assert(cuda::std::get<1>(p) == 4);
    cuda::std::get<0>(p) = 5;
    cuda::std::get<1>(p) = 6;
    assert(cuda::std::get<0>(p) == 5);
    assert(cuda::std::get<1>(p) == 6);
  }

  {
    static_assert(S().k == 1, "");
    static_assert(cuda::std::get<1>(getP()) == 4, "");
  }

  return 0;
}
