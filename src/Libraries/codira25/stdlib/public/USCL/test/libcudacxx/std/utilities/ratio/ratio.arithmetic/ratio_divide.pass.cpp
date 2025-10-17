/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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

// test ratio_divide

#include <uscl/std/ratio>

int main(int, char**)
{
  {
    typedef cuda::std::ratio<1, 1> R1;
    typedef cuda::std::ratio<1, 1> R2;
    typedef cuda::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == 1 && R::den == 1, "");
  }
  {
    typedef cuda::std::ratio<1, 2> R1;
    typedef cuda::std::ratio<1, 1> R2;
    typedef cuda::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == 1 && R::den == 2, "");
  }
  {
    typedef cuda::std::ratio<-1, 2> R1;
    typedef cuda::std::ratio<1, 1> R2;
    typedef cuda::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == -1 && R::den == 2, "");
  }
  {
    typedef cuda::std::ratio<1, -2> R1;
    typedef cuda::std::ratio<1, 1> R2;
    typedef cuda::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == -1 && R::den == 2, "");
  }
  {
    typedef cuda::std::ratio<1, 2> R1;
    typedef cuda::std::ratio<-1, 1> R2;
    typedef cuda::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == -1 && R::den == 2, "");
  }
  {
    typedef cuda::std::ratio<1, 2> R1;
    typedef cuda::std::ratio<1, -1> R2;
    typedef cuda::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == -1 && R::den == 2, "");
  }
  {
    typedef cuda::std::ratio<56987354, 467584654> R1;
    typedef cuda::std::ratio<544668, 22145> R2;
    typedef cuda::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == 630992477165LL && R::den == 127339199162436LL, "");
  }

  return 0;
}
