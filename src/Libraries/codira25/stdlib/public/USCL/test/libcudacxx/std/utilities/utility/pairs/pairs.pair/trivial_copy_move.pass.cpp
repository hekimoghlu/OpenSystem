/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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

// XFAIL: gcc-4

// <utility>

// template <class T1, class T2> struct pair

// pair(pair const&) = default;
// pair(pair&&) = default;

#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

struct Dummy
{
  Dummy(Dummy const&) = delete;
  Dummy(Dummy&&)      = default;
};

int main(int, char**)
{
  typedef cuda::std::pair<int, short> P;
  {
    static_assert(cuda::std::is_copy_constructible<P>::value, "");
    static_assert(cuda::std::is_trivially_copy_constructible<P>::value, "");
  }
  {
    static_assert(cuda::std::is_move_constructible<P>::value, "");
    static_assert(cuda::std::is_trivially_move_constructible<P>::value, "");
  }
  {
    using P1 = cuda::std::pair<Dummy, int>;
    static_assert(!cuda::std::is_copy_constructible<P1>::value, "");
    static_assert(!cuda::std::is_trivially_copy_constructible<P1>::value, "");
    static_assert(cuda::std::is_move_constructible<P1>::value, "");
    static_assert(cuda::std::is_trivially_move_constructible<P1>::value, "");
  }

  // extensions to ensure pair is trivially_copyable
  {
    static_assert(cuda::std::is_copy_assignable<P>::value, "");
    static_assert(cuda::std::is_trivially_copy_assignable<P>::value, "");
  }
  {
    static_assert(cuda::std::is_move_assignable<P>::value, "");
    static_assert(cuda::std::is_trivially_move_assignable<P>::value, "");
  }
  {
    static_assert(cuda::std::is_trivially_copyable<P>::value, "");
  }

  return 0;
}
