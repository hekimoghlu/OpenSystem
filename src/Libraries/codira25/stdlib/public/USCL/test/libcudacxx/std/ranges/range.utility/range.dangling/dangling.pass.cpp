/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

// cuda::std::ranges::dangling;

#include <uscl/std/concepts>
#include <uscl/std/ranges>
#include <uscl/std/type_traits>

#include "test_macros.h"

static_assert(cuda::std::is_empty_v<cuda::std::ranges::dangling>);

template <int>
struct S
{};
static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::ranges::dangling>);
static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::ranges::dangling, S<0>>);
static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::ranges::dangling, S<0>, S<1>>);
static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::ranges::dangling, S<0>, S<1>, S<2>>);

__host__ __device__ constexpr bool test_dangling()
{
  auto a = cuda::std::ranges::dangling();
  auto b = cuda::std::ranges::dangling(S<0>());
  auto c = cuda::std::ranges::dangling(S<0>(), S<1>());
  auto d = cuda::std::ranges::dangling(S<0>(), S<1>(), S<2>());
  unused(a);
  unused(b);
  unused(c);
  unused(d);
  return true;
}

int main(int, char**)
{
  static_assert(test_dangling());
  test_dangling();
  return 0;
}
