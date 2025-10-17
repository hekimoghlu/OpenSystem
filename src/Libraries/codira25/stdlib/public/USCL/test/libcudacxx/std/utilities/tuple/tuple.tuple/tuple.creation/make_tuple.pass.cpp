/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template<class... Types>
//   tuple<VTypes...> make_tuple(Types&&... t);

#include <uscl/std/cassert>
#include <uscl/std/functional>
#include <uscl/std/tuple>

#include "test_macros.h"

int main(int, char**)
{
  {
    int i                                 = 0;
    float j                               = 0;
    cuda::std::tuple<int, int&, float&> t = cuda::std::make_tuple(1, cuda::std::ref(i), cuda::std::ref(j));
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t) == 0);
    assert(cuda::std::get<2>(t) == 0);
    i = 2;
    j = 3.5;
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t) == 2);
    assert(cuda::std::get<2>(t) == 3.5);
    cuda::std::get<1>(t) = 0;
    cuda::std::get<2>(t) = 0;
    assert(i == 0);
    assert(j == 0);
  }
  {
    constexpr auto t1   = cuda::std::make_tuple(0, 1, 3.14);
    constexpr int i1    = cuda::std::get<1>(t1);
    constexpr double d1 = cuda::std::get<2>(t1);
    static_assert(i1 == 1, "");
    static_assert(d1 == 3.14, "");
  }

  return 0;
}
