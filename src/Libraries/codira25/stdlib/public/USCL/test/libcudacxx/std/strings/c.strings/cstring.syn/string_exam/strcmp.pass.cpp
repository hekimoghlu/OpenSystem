/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/cstring>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr void test_strcmp(const char* lhs, const char* rhs, int expected)
{
  const auto ret = cuda::std::strcmp(lhs, rhs);

  if (expected == 0)
  {
    assert(ret == 0);
  }
  else if (expected < 0)
  {
    assert(ret < 0);
  }
  else
  {
    assert(ret > 0);
  }
}

__host__ __device__ constexpr bool test()
{
  static_assert(
    cuda::std::
      is_same_v<int, decltype(cuda::std::strcmp(cuda::std::declval<const char*>(), cuda::std::declval<const char*>()))>);

  test_strcmp("", "", 0);

  test_strcmp("a", "", 1);
  test_strcmp("", "a", -1);

  test_strcmp("hi", "hi", 0);
  test_strcmp("hi", "ho", -1);
  test_strcmp("ho", "hi", 1);

  test_strcmp("abcde", "abcde", 0);
  test_strcmp("abcd1", "abcd0", 1);
  test_strcmp("abcd0", "abcd1", -1);
  test_strcmp("ab1de", "abcd0", -1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
