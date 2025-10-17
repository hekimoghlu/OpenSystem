/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
//     const typename tuple_element<I, cuda::std::pair<T1, T2> >::type&
//     get(const pair<T1, T2>&);

// UNSUPPORTED: msvc

#include <uscl/std/cassert>
#include <uscl/std/tuple>
#include <uscl/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::pair<int, short> P;
    const P p(3, static_cast<short>(4));
    assert(cuda::std::get<0>(p) == 3);
    assert(cuda::std::get<1>(p) == 4);
  }

  {
    typedef cuda::std::pair<int, short> P;
    constexpr P p1(3, static_cast<short>(4));
    static_assert(cuda::std::get<0>(p1) == 3, "");
    static_assert(cuda::std::get<1>(p1) == 4, "");
  }

  return 0;
}
