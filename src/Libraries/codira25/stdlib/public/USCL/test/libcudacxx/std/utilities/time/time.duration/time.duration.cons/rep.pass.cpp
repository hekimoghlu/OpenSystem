/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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

// template <class Rep2>
//   explicit duration(const Rep2& r);

#include "../../rep.h"

#include <uscl/std/cassert>
#include <uscl/std/chrono>

#include "test_macros.h"

template <class D, class R>
__host__ __device__ void test(R r)
{
  D d(r);
  assert(d.count() == r);
  constexpr D d2(R(2));
  static_assert(d2.count() == 2, "");
}

int main(int, char**)
{
  test<cuda::std::chrono::duration<int>>(5);
  test<cuda::std::chrono::duration<int, cuda::std::ratio<3, 2>>>(5);
  test<cuda::std::chrono::duration<Rep, cuda::std::ratio<3, 2>>>(Rep(3));
  test<cuda::std::chrono::duration<double, cuda::std::ratio<2, 3>>>(5.5);

  return 0;
}
