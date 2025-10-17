/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// sentinel() = default;
// constexpr explicit sentinel(sentinel_t<Base> end);
// constexpr sentinel(sentinel<!Const> s)
//   requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    cuda::std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView(buffer), 4);
    decltype(auto) sw1 = tv.end().base();
    static_assert(cuda::std::same_as<decltype(sw1), sentinel_wrapper<int*>>);
    assert(base(sw1) == buffer + 8);
    decltype(auto) sw2 = cuda::std::as_const(tv).end().base();
    static_assert(cuda::std::same_as<decltype(sw2), sentinel_wrapper<int*>>);
    assert(base(sw2) == buffer + 8);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
