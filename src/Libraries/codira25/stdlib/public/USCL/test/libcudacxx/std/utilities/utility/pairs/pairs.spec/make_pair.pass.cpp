/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> pair<V1, V2> make_pair(T1&&, T2&&);

#include <uscl/std/utility>
// cuda/std/memory not supported
// #include <uscl/std/memory>
#include <uscl/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::pair<int, short> P1;
    P1 p1 = cuda::std::make_pair(3, static_cast<short>(4));
    assert(p1.first == 3);
    assert(p1.second == 4);
  }

  // cuda/std/memory not supported
  /*
  {
      typedef cuda::std::pair<cuda::std::unique_ptr<int>, short> P1;
      P1 p1 = cuda::std::make_pair(cuda::std::unique_ptr<int>(new int(3)), static_cast<short>(4));
      assert(*p1.first == 3);
      assert(p1.second == 4);
  }
  {
      typedef cuda::std::pair<cuda::std::unique_ptr<int>, short> P1;
      P1 p1 = cuda::std::make_pair(nullptr, static_cast<short>(4));
      assert(p1.first == nullptr);
      assert(p1.second == 4);
  }
  */
  {
    typedef cuda::std::pair<int, short> P1;
    constexpr P1 p1 = cuda::std::make_pair(3, static_cast<short>(4));
    static_assert(p1.first == 3, "");
    static_assert(p1.second == 4, "");
  }

  return 0;
}
