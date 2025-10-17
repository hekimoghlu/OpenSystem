/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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

// template <class Rep1, class Period, class Rep2>
//   constexpr
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator*(const duration<Rep1, Period>& d, const Rep2& s);

// template <class Rep1, class Period, class Rep2>
//   constexpr
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator*(const Rep1& s, const duration<Rep2, Period>& d);

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::chrono::nanoseconds ns(3);
    ns = ns * 5;
    assert(ns.count() == 15);
    ns = 6 * ns;
    assert(ns.count() == 90);
  }
  {
    constexpr cuda::std::chrono::nanoseconds ns(3);
    constexpr cuda::std::chrono::nanoseconds ns2 = ns * 5;
    static_assert(ns2.count() == 15, "");
    constexpr cuda::std::chrono::nanoseconds ns3 = 6 * ns;
    static_assert(ns3.count() == 18, "");
  }

  return 0;
}
