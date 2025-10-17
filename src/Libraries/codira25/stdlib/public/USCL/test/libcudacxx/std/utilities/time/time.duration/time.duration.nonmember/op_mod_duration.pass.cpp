/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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

// duration

// template <class Rep1, class Period1, class Rep2, class Period2>
//   constexpr
//   typename common_type<duration<Rep1, Period1>, duration<Rep2, Period2>>::type
//   operator%(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::chrono::nanoseconds ns1(15);
    cuda::std::chrono::nanoseconds ns2(6);
    cuda::std::chrono::nanoseconds r = ns1 % ns2;
    assert(r.count() == 3);
  }
  {
    cuda::std::chrono::microseconds us1(15);
    cuda::std::chrono::nanoseconds ns2(28);
    cuda::std::chrono::nanoseconds r = us1 % ns2;
    assert(r.count() == 20);
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s1(6);
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s2(3);
    cuda::std::chrono::duration<int, cuda::std::ratio<1, 15>> r = s1 % s2;
    assert(r.count() == 24);
  }
  {
    constexpr cuda::std::chrono::nanoseconds ns1(15);
    constexpr cuda::std::chrono::nanoseconds ns2(6);
    constexpr cuda::std::chrono::nanoseconds r = ns1 % ns2;
    static_assert(r.count() == 3, "");
  }
  {
    constexpr cuda::std::chrono::microseconds us1(15);
    constexpr cuda::std::chrono::nanoseconds ns2(28);
    constexpr cuda::std::chrono::nanoseconds r = us1 % ns2;
    static_assert(r.count() == 20, "");
  }
  {
    constexpr cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s1(6);
    constexpr cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s2(3);
    constexpr cuda::std::chrono::duration<int, cuda::std::ratio<1, 15>> r = s1 % s2;
    static_assert(r.count() == 24, "");
  }

  return 0;
}
