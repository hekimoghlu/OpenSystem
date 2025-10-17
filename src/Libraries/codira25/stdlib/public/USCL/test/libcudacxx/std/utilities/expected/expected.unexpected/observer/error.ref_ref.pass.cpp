/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr E&& error() && noexcept;

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/utility>

#include "test_macros.h"

template <class T, class = void>
constexpr bool ErrorNoexcept = false;

template <class T>
constexpr bool ErrorNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T&&>().error())>> =
  noexcept(cuda::std::declval<T&&>().error());

static_assert(!ErrorNoexcept<int>, "");
static_assert(ErrorNoexcept<cuda::std::unexpected<int>>, "");

__host__ __device__ constexpr bool test()
{
  cuda::std::unexpected<int> unex(5);
  decltype(auto) i = cuda::std::move(unex).error();
  static_assert(cuda::std::same_as<decltype(i), int&&>, "");
  assert(i == 5);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
