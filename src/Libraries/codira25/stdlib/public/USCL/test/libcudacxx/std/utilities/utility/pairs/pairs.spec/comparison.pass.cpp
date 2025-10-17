/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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

// template <class T1, class T2> struct pair

// template <class T1, class T2> bool operator==(const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator!=(const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator< (const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator> (const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator>=(const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator<=(const pair<T1,T2>&, const pair<T1,T2>&);

#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::pair<int, short> P;
    P p1(3, static_cast<short>(4));
    P p2(3, static_cast<short>(4));
    assert((p1 == p2));
    assert(!(p1 != p2));
    assert(!(p1 < p2));
    assert((p1 <= p2));
    assert(!(p1 > p2));
    assert((p1 >= p2));
  }
  {
    typedef cuda::std::pair<int, short> P;
    P p1(2, static_cast<short>(4));
    P p2(3, static_cast<short>(4));
    assert(!(p1 == p2));
    assert((p1 != p2));
    assert((p1 < p2));
    assert((p1 <= p2));
    assert(!(p1 > p2));
    assert(!(p1 >= p2));
  }
  {
    typedef cuda::std::pair<int, short> P;
    P p1(3, static_cast<short>(2));
    P p2(3, static_cast<short>(4));
    assert(!(p1 == p2));
    assert((p1 != p2));
    assert((p1 < p2));
    assert((p1 <= p2));
    assert(!(p1 > p2));
    assert(!(p1 >= p2));
  }
  {
    typedef cuda::std::pair<int, short> P;
    P p1(3, static_cast<short>(4));
    P p2(2, static_cast<short>(4));
    assert(!(p1 == p2));
    assert((p1 != p2));
    assert(!(p1 < p2));
    assert(!(p1 <= p2));
    assert((p1 > p2));
    assert((p1 >= p2));
  }
  {
    typedef cuda::std::pair<int, short> P;
    P p1(3, static_cast<short>(4));
    P p2(3, static_cast<short>(2));
    assert(!(p1 == p2));
    assert((p1 != p2));
    assert(!(p1 < p2));
    assert(!(p1 <= p2));
    assert((p1 > p2));
    assert((p1 >= p2));
  }

  {
    typedef cuda::std::pair<int, short> P;
    constexpr P p1(3, static_cast<short>(4));
    constexpr P p2(3, static_cast<short>(2));
    static_assert(!(p1 == p2), "");
    static_assert((p1 != p2), "");
    static_assert(!(p1 < p2), "");
    static_assert(!(p1 <= p2), "");
    static_assert((p1 > p2), "");
    static_assert((p1 >= p2), "");
  }

  return 0;
}
