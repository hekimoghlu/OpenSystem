/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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

// <cuda/std/tuple>

// constexpr unspecified ignore;

#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "test_macros.h"

static_assert(cuda::std::is_trivially_default_constructible<decltype(cuda::std::ignore)>::value
                && cuda::std::is_empty<decltype(cuda::std::ignore)>::value,
              "");

// constexpr variables are unavailable before 11.3
[[nodiscard]] __host__ __device__ constexpr int test_nodiscard()
{
  return 8294;
}

__host__ __device__ constexpr bool test()
{
  {
    auto& ignore_v = cuda::std::ignore;
    unused(ignore_v);
  }

  { // Test that cuda::std::ignore provides converting assignment.
    auto& res = (cuda::std::ignore = 42);
    static_assert(noexcept(res = (cuda::std::ignore = 42)), "");
    assert(&res == &cuda::std::ignore);
  }
  { // Test bit-field binding.
    struct S
    {
      unsigned int bf : 3;
    };
    S s{3};
    auto& res = (cuda::std::ignore = s.bf);
    assert(&res == &cuda::std::ignore);
  }
  { // Test that cuda::std::ignore provides constexpr copy/move constructors
    auto copy  = cuda::std::ignore;
    auto moved = cuda::std::move(copy);
    unused(moved);
  }
  { // Test that cuda::std::ignore provides constexpr copy/move assignment
    auto copy  = cuda::std::ignore;
    copy       = cuda::std::ignore;
    auto moved = cuda::std::ignore;
    moved      = cuda::std::move(copy);
    unused(moved);
  }

  {
    cuda::std::ignore = test_nodiscard();
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
