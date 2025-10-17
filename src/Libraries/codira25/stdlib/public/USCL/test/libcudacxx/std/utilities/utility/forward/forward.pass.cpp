/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test forward

#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

struct A
{};

__host__ __device__ A source() noexcept
{
  return A();
}
__host__ __device__ const A csource() noexcept
{
  return A();
}

__host__ __device__ constexpr bool test_constexpr_forward()
{
  int x        = 42;
  const int cx = 101;
  return cuda::std::forward<int&>(x) == 42 && cuda::std::forward<int>(x) == 42
      && cuda::std::forward<const int&>(x) == 42 && cuda::std::forward<const int>(x) == 42
      && cuda::std::forward<int&&>(x) == 42 && cuda::std::forward<const int&&>(x) == 42
      && cuda::std::forward<const int&>(cx) == 101 && cuda::std::forward<const int>(cx) == 101;
}

int main(int, char**)
{
  [[maybe_unused]] A a;
  [[maybe_unused]] const A ca = A();

  static_assert(cuda::std::is_same<decltype(cuda::std::forward<A&>(a)), A&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<A>(a)), A&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<A>(source())), A&&>::value, "");
  static_assert(noexcept(cuda::std::forward<A&>(a)));
  static_assert(noexcept(cuda::std::forward<A>(a)));
  static_assert(noexcept(cuda::std::forward<A>(source())));

  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A&>(a)), const A&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(a)), const A&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(source())), const A&&>::value, "");
  static_assert(noexcept(cuda::std::forward<const A&>(a)));
  static_assert(noexcept(cuda::std::forward<const A>(a)));
  static_assert(noexcept(cuda::std::forward<const A>(source())));

  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A&>(ca)), const A&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(ca)), const A&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(csource())), const A&&>::value, "");
  static_assert(noexcept(cuda::std::forward<const A&>(ca)));
  static_assert(noexcept(cuda::std::forward<const A>(ca)));
  static_assert(noexcept(cuda::std::forward<const A>(csource())));

  {
    constexpr int i2 = cuda::std::forward<int>(42);
    static_assert(cuda::std::forward<int>(42) == 42, "");
    static_assert(cuda::std::forward<const int&>(i2) == 42, "");
    static_assert(test_constexpr_forward(), "");
  }

  return 0;
}
