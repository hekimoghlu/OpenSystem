/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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

// tuple(const tuple& u) = default;

#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "test_macros.h"

struct Empty
{};

int main(int, char**)
{
  {
    using T = cuda::std::tuple<>;
    T t0;
    T t = t0;
    unused(t); // Prevent unused warning
  }
  {
    using T = cuda::std::tuple<int>;
    T t0(2);
    T t = t0;
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    using T = cuda::std::tuple<int, char>;
    T t0(2, 'a');
    T t = t0;
    assert(cuda::std::get<0>(t) == 2);
    assert(cuda::std::get<1>(t) == 'a');
  }
  // cuda::std::string not supported
  /*
  {
      using T = cuda::std::tuple<int, char, cuda::std::string>;
      const T t0(2, 'a', "some text");
      T t = t0;
      assert(cuda::std::get<0>(t) == 2);
      assert(cuda::std::get<1>(t) == 'a');
      assert(cuda::std::get<2>(t) == "some text");
  }
  */
  {
    using T = cuda::std::tuple<int>;
    constexpr T t0(2);
    constexpr T t = t0;
    static_assert(cuda::std::get<0>(t) == 2, "");
  }
  {
    using T = cuda::std::tuple<Empty>;
    constexpr T t0;
    constexpr T t                      = t0;
    [[maybe_unused]] constexpr Empty e = cuda::std::get<0>(t);
  }

  return 0;
}
