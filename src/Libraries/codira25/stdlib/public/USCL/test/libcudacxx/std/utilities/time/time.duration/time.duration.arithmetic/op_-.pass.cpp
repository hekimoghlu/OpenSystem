/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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

// constexpr common_type_t<duration> operator-() const;

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(set_but_not_used)

int main(int, char**)
{
  {
    const cuda::std::chrono::minutes m(3);
    cuda::std::chrono::minutes m2 = -m;
    assert(m2.count() == -m.count());
  }
  {
    constexpr cuda::std::chrono::minutes m(3);
    constexpr cuda::std::chrono::minutes m2 = -m;
    static_assert(m2.count() == -m.count(), "");
  }

  // P0548
  {
    typedef cuda::std::chrono::duration<int, cuda::std::ratio<10, 10>> D10;
    typedef cuda::std::chrono::duration<int, cuda::std::ratio<1, 1>> D1;
    D10 zero(0);
    D10 one(1);
    static_assert((cuda::std::is_same<decltype(-one), decltype(zero - one)>::value), "");
    static_assert((cuda::std::is_same<decltype(zero - one), D1>::value), "");
    static_assert((cuda::std::is_same<decltype(-one), D1>::value), "");
    static_assert((cuda::std::is_same<decltype(+one), D1>::value), "");
    unused(zero);
    unused(one);
  }

  return 0;
}
