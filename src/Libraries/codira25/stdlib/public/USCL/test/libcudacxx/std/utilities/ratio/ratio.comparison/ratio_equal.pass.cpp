/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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

// test ratio_equal

#include <uscl/std/ratio>

#include "test_macros.h"

template <class Rat1, class Rat2, bool result>
__host__ __device__ void test()
{
  static_assert((result == cuda::std::ratio_equal<Rat1, Rat2>::value), "");
  static_assert((result == cuda::std::ratio_equal_v<Rat1, Rat2>), "");
}

int main(int, char**)
{
  {
    typedef cuda::std::ratio<1, 1> R1;
    typedef cuda::std::ratio<1, 1> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
    typedef cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<1, 1> R1;
    typedef cuda::std::ratio<1, -1> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
    typedef cuda::std::ratio<1, -0x7FFFFFFFFFFFFFFFLL> R2;
    test<R1, R2, false>();
  }

  return 0;
}
