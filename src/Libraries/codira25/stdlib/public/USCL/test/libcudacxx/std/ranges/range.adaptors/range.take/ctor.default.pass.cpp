/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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

// take_view() requires default_initializable<V> = default;

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE constexpr int buff[8] = {1, 2, 3, 4, 5, 6, 7, 8};

struct DefaultConstructible : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr DefaultConstructible()
      : begin_(buff)
      , end_(buff + 8)
  {}
  __host__ __device__ constexpr int const* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr int const* end() const
  {
    return end_;
  }

private:
  int const* begin_;
  int const* end_;
};

struct NonDefaultConstructible : cuda::std::ranges::view_base
{
  NonDefaultConstructible() = delete;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::take_view<DefaultConstructible> tv;
    assert(tv.begin() == buff);
    assert(tv.size() == 0);
  }

  // Test SFINAE-friendliness
  {
    static_assert(cuda::std::is_default_constructible_v<cuda::std::ranges::take_view<DefaultConstructible>>);
    static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_view<NonDefaultConstructible>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
