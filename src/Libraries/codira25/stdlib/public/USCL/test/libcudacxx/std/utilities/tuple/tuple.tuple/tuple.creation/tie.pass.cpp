/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
//   tuple<Types&...> tie(Types&... t);

#include <uscl/std/tuple>

// cuda::std::string not supported
// #include <uscl/std/string>
#include <uscl/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test_tie_constexpr()
{
  {
    int i         = 42;
    double f      = 1.1;
    using ExpectT = cuda::std::tuple<int&, decltype(cuda::std::ignore)&, double&>;
    auto res      = cuda::std::tie(i, cuda::std::ignore, f);
    static_assert(cuda::std::is_same<ExpectT, decltype(res)>::value, "");
    assert(&cuda::std::get<0>(res) == &i);
    assert(&cuda::std::get<1>(res) == &cuda::std::ignore);
    assert(&cuda::std::get<2>(res) == &f);
    // FIXME: If/when tuple gets constexpr assignment
    // res = cuda::std::make_tuple(101, nullptr, -1.0);
  }
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 int i          = 0;
                 const char* _s = "C++";
                 // cuda::std::string not supported
                 // cuda::std::string s;
                 const char* s;
                 cuda::std::tie(i, cuda::std::ignore, s) = cuda::std::make_tuple(42, 3.14, _s);
                 assert(i == 42);
                 assert(s == _s);
               }))
  {
    static constexpr int i                                  = 42;
    static constexpr double f                               = 1.1;
    constexpr cuda::std::tuple<const int&, const double&> t = cuda::std::tie(i, f);
    static_assert(cuda::std::get<0>(t) == 42, "");
    static_assert(cuda::std::get<1>(t) == 1.1, "");
  }
  {
    static_assert(test_tie_constexpr(), "");
  }

  return 0;
}
