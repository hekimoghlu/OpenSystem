/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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

// constexpr const T& operator*() const & noexcept;
// constexpr T& operator*() & noexcept;
// constexpr T&& operator*() && noexcept;
// constexpr const T&& operator*() const && noexcept;

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

static_assert(DerefNoexcept<cuda::std::expected<int, int>&>, "");
static_assert(DerefNoexcept<const cuda::std::expected<int, int>&>, "");
static_assert(DerefNoexcept<cuda::std::expected<int, int>&&>, "");
static_assert(DerefNoexcept<const cuda::std::expected<int, int>&&>, "");

__host__ __device__ constexpr bool test()
{
  // non-const &
  {
    cuda::std::expected<int, int> e(5);
    decltype(auto) x = *e;
    static_assert(cuda::std::same_as<decltype(x), int&>, "");
    assert(&x == &(e.value()));
    assert(x == 5);
  }

  // const &
  {
    const cuda::std::expected<int, int> e(5);
    decltype(auto) x = *e;
    static_assert(cuda::std::same_as<decltype(x), const int&>, "");
    assert(&x == &(e.value()));
    assert(x == 5);
  }

  // non-const &&
  {
    cuda::std::expected<int, int> e(5);
    decltype(auto) x = *cuda::std::move(e);
    static_assert(cuda::std::same_as<decltype(x), int&&>, "");
    assert(&x == &(e.value()));
    assert(x == 5);
  }

  // const &&
  {
    const cuda::std::expected<int, int> e(5);
    decltype(auto) x = *cuda::std::move(e);
    static_assert(cuda::std::same_as<decltype(x), const int&&>, "");
    assert(&x == &(e.value()));
    assert(x == 5);
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
