/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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

// constexpr expected() noexcept;

#include <uscl/std/cassert>
#include <uscl/std/expected>
#include <uscl/std/type_traits>

#include "test_macros.h"

// Test noexcept

struct NoDefaultCtor
{
  __host__ __device__ constexpr NoDefaultCtor() = delete;
};

static_assert(cuda::std::is_nothrow_default_constructible_v<cuda::std::expected<void, int>>, "");
static_assert(cuda::std::is_nothrow_default_constructible_v<cuda::std::expected<void, NoDefaultCtor>>, "");

struct MyInt
{
  int i;
#if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const MyInt&, const MyInt&) = default;
#else
  __host__ __device__ friend constexpr bool operator==(const MyInt& lhs, const MyInt& rhs) noexcept
  {
    return lhs.i == rhs.i;
  }
  __host__ __device__ friend constexpr bool operator!=(const MyInt& lhs, const MyInt& rhs) noexcept
  {
    return lhs.i != rhs.i;
  }
#endif // TEST_STD_VER > 2017
};

__host__ __device__ constexpr bool test()
{
  // default constructible
  {
    cuda::std::expected<void, int> e;
    assert(e.has_value());
  }

  // non-default constructible
  {
    cuda::std::expected<void, NoDefaultCtor> e;
    assert(e.has_value());
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
