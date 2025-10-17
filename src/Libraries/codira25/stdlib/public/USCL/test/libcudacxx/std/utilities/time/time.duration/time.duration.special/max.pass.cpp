/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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

// static constexpr duration max(); // noexcept after C++17

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/limits>

#include "../../rep.h"
#include "test_macros.h"

template <class D>
__host__ __device__ void test()
{
  static_assert(noexcept(cuda::std::chrono::duration_values<typename D::rep>::max()));
#if TEST_STD_VER > 2017
  static_assert(noexcept(cuda::std::chrono::duration_values<typename D::rep>::max()));
#endif
  {
    typedef typename D::rep Rep;
    Rep max_rep = cuda::std::chrono::duration_values<Rep>::max();
    assert(D::max().count() == max_rep);
  }
  {
    typedef typename D::rep Rep;
    constexpr Rep max_rep = cuda::std::chrono::duration_values<Rep>::max();
    static_assert(D::max().count() == max_rep, "");
  }
}

int main(int, char**)
{
  test<cuda::std::chrono::duration<int>>();
  test<cuda::std::chrono::duration<Rep>>();

  return 0;
}
