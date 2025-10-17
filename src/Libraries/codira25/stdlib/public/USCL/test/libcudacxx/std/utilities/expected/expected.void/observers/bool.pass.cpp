/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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

// constexpr explicit operator bool() const noexcept;

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

// Test noexcept
template <class T, class = void>
constexpr bool OpBoolNoexcept = false;

template <class T>
constexpr bool OpBoolNoexcept<T, cuda::std::void_t<decltype(static_cast<bool>(cuda::std::declval<T>()))>> =
  noexcept(static_cast<bool>(cuda::std::declval<T>()));

struct Foo
{};
static_assert(!OpBoolNoexcept<Foo>, "");

static_assert(OpBoolNoexcept<cuda::std::expected<void, int>>, "");
static_assert(OpBoolNoexcept<const cuda::std::expected<void, int>>, "");

// Test explicit
static_assert(!cuda::std::is_convertible_v<cuda::std::expected<void, int>, bool>, "");

__host__ __device__ constexpr bool test()
{
  // has_value
  {
    const cuda::std::expected<void, int> e;
    assert(static_cast<bool>(e));
  }

  // !has_value
  {
    const cuda::std::expected<void, int> e(cuda::std::unexpect, 5);
    assert(!static_cast<bool>(e));
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
