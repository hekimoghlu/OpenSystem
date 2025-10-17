/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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

// struct unreachable_sentinel_t;
// inline constexpr unreachable_sentinel_t unreachable_sentinel;

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/iterator>
#include <uscl/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  static_assert(cuda::std::is_empty_v<cuda::std::unreachable_sentinel_t>);
  static_assert(cuda::std::semiregular<cuda::std::unreachable_sentinel_t>);

  static_assert(cuda::std::same_as<decltype(cuda::std::unreachable_sentinel), const cuda::std::unreachable_sentinel_t>);

  auto sentinel = cuda::std::unreachable_sentinel;
  int i         = 42;
  assert(i != sentinel);
  assert(sentinel != i);
  assert(!(i == sentinel));
  assert(!(sentinel == i));

  assert(&i != sentinel);
  assert(sentinel != &i);
  assert(!(&i == sentinel));
  assert(!(sentinel == &i));

  int* p = nullptr;
  assert(p != sentinel);
  assert(sentinel != p);
  assert(!(p == sentinel));
  assert(!(sentinel == p));

  static_assert(cuda::std::__weakly_equality_comparable_with<cuda::std::unreachable_sentinel_t, int>);
  static_assert(cuda::std::__weakly_equality_comparable_with<cuda::std::unreachable_sentinel_t, int*>);
#if !TEST_COMPILER(GCC, <, 12) || TEST_STD_VER < 2020 // gcc 10 has an issue
                                                      // with void
  static_assert(!cuda::std::__weakly_equality_comparable_with<cuda::std::unreachable_sentinel_t, void*>);
#endif // !TEST_COMPILER(GCC, <, 12)  || TEST_STD_VER < 2020
  static_assert(noexcept(sentinel == p));
  static_assert(noexcept(sentinel != p));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
