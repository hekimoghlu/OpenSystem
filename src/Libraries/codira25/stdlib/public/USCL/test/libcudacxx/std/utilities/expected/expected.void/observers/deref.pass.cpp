/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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

// constexpr void operator*() const & noexcept;

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

// Test noexcept
template <class T, class = void>
constexpr bool DerefNoexcept = false;

template <class T>
constexpr bool DerefNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T>().operator*())>> =
  noexcept(cuda::std::declval<T>().operator*());

static_assert(!DerefNoexcept<int>, "");

static_assert(DerefNoexcept<cuda::std::expected<void, int>>, "");

__host__ __device__ constexpr bool test()
{
  const cuda::std::expected<void, int> e;
  *e;
  static_assert(cuda::std::is_same_v<decltype(*e), void>, "");

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
