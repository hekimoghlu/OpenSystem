/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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

// constexpr optional() noexcept;

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

template <class Opt>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::is_nothrow_default_constructible<Opt>::value, "");
  static_assert(cuda::std::is_trivially_destructible<Opt>::value
                  == cuda::std::is_trivially_destructible<typename Opt::value_type>::value,
                "");
  {
    Opt opt{};
    assert(static_cast<bool>(opt) == false);
  }
  {
    const Opt opt{};
    assert(static_cast<bool>(opt) == false);
  }
}

__host__ __device__ constexpr bool test()
{
  test<optional<int>>();
  test<optional<int*>>();
  test<optional<ImplicitTypes::NoCtors>>();
  test<optional<NonTrivialTypes::NoCtors>>();
  test<optional<NonConstexprTypes::NoCtors>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  test<optional<NonLiteralTypes::NoCtors>>();

  return 0;
}
