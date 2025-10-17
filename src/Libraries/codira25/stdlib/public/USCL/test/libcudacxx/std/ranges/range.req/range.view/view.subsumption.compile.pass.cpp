/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

// UNSUPPORTED: c++17
// UNSUPPORTED: msvc-19.16
// XFAIL: nvcc-12.0 || nvcc-12.1 || nvcc-12.2 || nvcc-12.3
// nvbug 3885350

// <ranges>

// template<class T>
// concept view = ...;

#include <uscl/std/ranges>

#include "test_macros.h"

struct View : cuda::std::ranges::view_base
{
  View()                  = default;
  View(View&&)            = default;
  View& operator=(View&&) = default;
  __host__ __device__ friend int* begin(View&);
  __host__ __device__ friend int* begin(View const&);
  __host__ __device__ friend int* end(View&);
  __host__ __device__ friend int* end(View const&);
};

namespace subsume_range
{
template <cuda::std::ranges::view>
__host__ __device__ constexpr bool test()
{
  return true;
}
template <cuda::std::ranges::range>
__host__ __device__ constexpr bool test()
{
  return false;
}
static_assert(test<View>(), "");
} // namespace subsume_range

#ifndef __NVCOMPILER // nvbug 3885350
namespace subsume_movable
{
template <cuda::std::ranges::view>
__host__ __device__ constexpr bool test()
{
  return true;
}
template <cuda::std::movable>
__host__ __device__ constexpr bool test()
{
  return false;
}
static_assert(test<View>(), "");
} // namespace subsume_movable
#endif

int main(int, char**)
{
  return 0;
}
