/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 24, 2023.
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

// template <class Clock, class Duration1, class Rep2, class Period2>
//   time_point<Clock, typename common_type<Duration1, duration<Rep2, Period2>>::type>
//   operator-(const time_point<Clock, Duration1>& lhs, const duration<Rep2, Period2>& rhs);

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

template <class D>
__host__ __device__ void test2739() // LWG2739
{
  typedef cuda::std::chrono::time_point<cuda::std::chrono::system_clock> TimePoint;
  typedef cuda::std::chrono::duration<D> Dur;
  const Dur d(5);
  TimePoint t0 = cuda::std::chrono::system_clock::from_time_t(200);
  TimePoint t1 = t0 - d;
  assert(t1 < t0);
}

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::milliseconds Duration1;
  typedef cuda::std::chrono::microseconds Duration2;
  {
    cuda::std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    cuda::std::chrono::time_point<Clock, Duration2> t2 = t1 - Duration2(5);
    assert(t2.time_since_epoch() == Duration2(2995));
  }
  {
    constexpr cuda::std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    constexpr cuda::std::chrono::time_point<Clock, Duration2> t2 = t1 - Duration2(5);
    static_assert(t2.time_since_epoch() == Duration2(2995), "");
  }
  test2739<int32_t>();
  test2739<uint32_t>();

  return 0;
}
