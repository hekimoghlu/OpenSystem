/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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

// <cuda/std/chrono>

// time_point

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator< (const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator> (const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator<=(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator>=(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::milliseconds Duration1;
  typedef cuda::std::chrono::microseconds Duration2;
  typedef cuda::std::chrono::time_point<Clock, Duration1> T1;
  typedef cuda::std::chrono::time_point<Clock, Duration2> T2;

  {
    T1 t1(Duration1(3));
    T1 t2(Duration1(3));
    assert(!(t1 < t2));
    assert(!(t1 > t2));
    assert((t1 <= t2));
    assert((t1 >= t2));
  }
  {
    T1 t1(Duration1(3));
    T1 t2(Duration1(4));
    assert((t1 < t2));
    assert(!(t1 > t2));
    assert((t1 <= t2));
    assert(!(t1 >= t2));
  }
  {
    T1 t1(Duration1(3));
    T2 t2(Duration2(3000));
    assert(!(t1 < t2));
    assert(!(t1 > t2));
    assert((t1 <= t2));
    assert((t1 >= t2));
  }
  {
    T1 t1(Duration1(3));
    T2 t2(Duration2(3001));
    assert((t1 < t2));
    assert(!(t1 > t2));
    assert((t1 <= t2));
    assert(!(t1 >= t2));
  }

  {
    constexpr T1 t1(Duration1(3));
    constexpr T1 t2(Duration1(3));
    static_assert(!(t1 < t2), "");
    static_assert(!(t1 > t2), "");
    static_assert((t1 <= t2), "");
    static_assert((t1 >= t2), "");
  }
  {
    constexpr T1 t1(Duration1(3));
    constexpr T1 t2(Duration1(4));
    static_assert((t1 < t2), "");
    static_assert(!(t1 > t2), "");
    static_assert((t1 <= t2), "");
    static_assert(!(t1 >= t2), "");
  }
  {
    constexpr T1 t1(Duration1(3));
    constexpr T2 t2(Duration2(3000));
    static_assert(!(t1 < t2), "");
    static_assert(!(t1 > t2), "");
    static_assert((t1 <= t2), "");
    static_assert((t1 >= t2), "");
  }
  {
    constexpr T1 t1(Duration1(3));
    constexpr T2 t2(Duration2(3001));
    static_assert((t1 < t2), "");
    static_assert(!(t1 > t2), "");
    static_assert((t1 <= t2), "");
    static_assert(!(t1 >= t2), "");
  }

  return 0;
}
