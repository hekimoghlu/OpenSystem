/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

// Older Clangs do not support the C++20 feature to constrain destructors

// constexpr ~expected();
//
// Effects: If has_value() is true, destroys val, otherwise destroys unex.
//
// Remarks: If is_trivially_destructible_v<T> is true, and is_trivially_destructible_v<E> is true,
// then this destructor is a trivial destructor.

#include <uscl/std/cassert>
#include <uscl/std/expected>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

// Test Remarks: If is_trivially_destructible_v<T> is true, and is_trivially_destructible_v<E> is true,
// then this destructor is a trivial destructor.
struct NonTrivial
{
  __host__ __device__ ~NonTrivial() {}
};

static_assert(cuda::std::is_trivially_destructible_v<cuda::std::expected<int, int>>, "");
static_assert(!cuda::std::is_trivially_destructible_v<cuda::std::expected<NonTrivial, int>>, "");
static_assert(!cuda::std::is_trivially_destructible_v<cuda::std::expected<int, NonTrivial>>, "");
static_assert(!cuda::std::is_trivially_destructible_v<cuda::std::expected<NonTrivial, NonTrivial>>, "");

struct TrackedDestroy
{
  bool& destroyed;
  __host__ __device__ constexpr TrackedDestroy(bool& b)
      : destroyed(b)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~TrackedDestroy()
  {
    destroyed = true;
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // has value
  {
    bool valueDestroyed = false;
    {
      cuda::std::expected<TrackedDestroy, TrackedDestroy> e(cuda::std::in_place, valueDestroyed);
      unused(e);
    }
    assert(valueDestroyed);
  }

  // has error
  {
    bool errorDestroyed = false;
    {
      cuda::std::expected<TrackedDestroy, TrackedDestroy> e(cuda::std::unexpect, errorDestroyed);
      unused(e);
    }
    assert(errorDestroyed);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
