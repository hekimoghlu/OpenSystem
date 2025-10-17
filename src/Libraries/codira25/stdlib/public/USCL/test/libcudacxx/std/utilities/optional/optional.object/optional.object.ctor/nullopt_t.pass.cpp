/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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

// <cuda/std/optional>

// constexpr optional(nullopt_t) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::nullopt;
using cuda::std::nullopt_t;
using cuda::std::optional;

template <class T>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::is_nothrow_constructible<optional<T>, nullopt_t&>::value, "");
  static_assert(
    cuda::std::is_trivially_destructible<optional<T>>::value == cuda::std::is_trivially_destructible<T>::value, "");
  {
    optional<T> opt{nullopt};
    assert(static_cast<bool>(opt) == false);
  }
  {
    const optional<T> opt{nullopt};
    assert(static_cast<bool>(opt) == false);
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<int*>();
  test<const int>();

  test<ImplicitTypes::NoCtors>();
  test<NonTrivialTypes::NoCtors>();
  test<NonConstexprTypes::NoCtors>();

#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

__global__ void test_global_visibility()
{
  cuda::std::optional<int> meow{cuda::std::nullopt};
  unused(meow);
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
