/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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

// duration& operator%=(const duration& rhs)

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

__host__ __device__ constexpr bool test_constexpr()
{
  cuda::std::chrono::microseconds us1(11);
  cuda::std::chrono::microseconds us2(3);
  us1 %= us2;
  return us1.count() == 2;
}

int main(int, char**)
{
  {
    cuda::std::chrono::microseconds us1(11);
    cuda::std::chrono::microseconds us2(3);
    us1 %= us2;
    assert(us1.count() == 2);
    us1 %= cuda::std::chrono::milliseconds(3);
    assert(us1.count() == 2);
  }

  static_assert(test_constexpr(), "");

  return 0;
}
