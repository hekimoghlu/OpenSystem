/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// sentinel() = default;

#include <uscl/std/cassert>
#include <uscl/std/ranges>

struct Sent
{
  bool b; // deliberately uninitialised

  __host__ __device__ friend constexpr bool operator==(int*, const Sent& s)
  {
    return s.b;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator==(const Sent& s, int*)
  {
    return s.b;
  }
  __host__ __device__ friend constexpr bool operator!=(int*, const Sent& s)
  {
    return !s.b;
  }
  __host__ __device__ friend constexpr bool operator!=(const Sent& s, int*)
  {
    return !s.b;
  }
#endif // TEST_STD_VER <= 2017
};

struct Range : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const
  {
    return nullptr;
  }
  __host__ __device__ Sent end()
  {
    return Sent{};
  }
};

__host__ __device__ constexpr bool test()
{
  {
    using R        = cuda::std::ranges::take_while_view<Range, bool (*)(int)>;
    using Sentinel = cuda::std::ranges::sentinel_t<R>;

    Sentinel s1 = {};
    assert(!s1.base().b);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
