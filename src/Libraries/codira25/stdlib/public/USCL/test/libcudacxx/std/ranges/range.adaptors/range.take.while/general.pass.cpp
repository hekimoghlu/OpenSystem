/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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

// Some basic examples of how take_while_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

// #include <uscl/std/algorithm>
#include <uscl/std/cassert>
#include <uscl/std/ranges>

template <class Range, class Expected>
__host__ __device__ constexpr bool equal(Range&& range, Expected&& expected)
{
  auto irange    = range.begin();
  auto iexpected = cuda::std::begin(expected);
  for (; irange != range.end(); ++irange, ++iexpected)
  {
    if (*irange != *iexpected)
    {
      return false;
    }
  }
  return true;
}

int main(int, char**)
{
  {
    auto input = {0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0};
    auto small = [](const int x) noexcept {
      return x < 5;
    };
    auto small_ints = input | cuda::std::views::take_while(small);
    auto expected   = {0, 1, 2, 3, 4};
    assert(equal(small_ints, expected));
  }
  return 0;
}
