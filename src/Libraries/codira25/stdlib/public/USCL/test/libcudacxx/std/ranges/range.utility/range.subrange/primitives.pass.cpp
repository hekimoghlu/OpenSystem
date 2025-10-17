/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_iterators.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buff[] = {1, 2, 3, 4, 5};

  {
    cuda::std::ranges::subrange<MoveOnlyForwardIter, int*> a(MoveOnlyForwardIter(buff), buff + 5, 5);
    assert(base(a.begin()) == buff);
    assert(!a.empty());
    assert(a.size() == 5);
  }

  {
    cuda::std::ranges::subrange<ForwardIter> b(ForwardIter(nullptr), ForwardIter(nullptr));
    assert(b.empty());
  }

  {
    cuda::std::ranges::subrange<ForwardIter> c{ForwardIter(buff), ForwardIter(buff)};
    assert(c.empty());
  }

  {
    cuda::std::ranges::subrange<ForwardIter> d(ForwardIter(buff), ForwardIter(buff + 1));
    assert(!d.empty());
  }

  {
    bool minusWasCalled = false;
    SizedSentinelForwardIter beg(buff, &minusWasCalled), end(buff + 5, &minusWasCalled);
    cuda::std::ranges::subrange<SizedSentinelForwardIter> e(beg, end, 5);
    assert(!e.empty());

    // Make sure that operator- is used to calculate size when possible.
    minusWasCalled = false;
    assert(e.size() == 5);
    assert(minusWasCalled);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
