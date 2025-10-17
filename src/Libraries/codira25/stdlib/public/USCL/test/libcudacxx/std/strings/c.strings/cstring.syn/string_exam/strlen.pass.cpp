/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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

__host__ __device__ constexpr void test_strlen(const char* str, cuda::std::size_t expected)
{
  const auto ret = cuda::std::strlen(str);
  assert(ret == expected);
}

__host__ __device__ constexpr bool test()
{
  static_assert(
    cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::std::strlen(cuda::std::declval<const char*>()))>);

  test_strlen("", 0);
  test_strlen("a", 1);
  test_strlen("hi", 2);
  test_strlen("asdfgh", 6);
  test_strlen("asdfasdfasdfgweyr", 17);
  test_strlen("asdfasdfasdfgweyr1239859102384", 30);

  test_strlen("\0\0", 0);
  test_strlen("a\0", 1);
  test_strlen("h\0i", 1);
  test_strlen("asdf\0g\0h", 4);
  test_strlen("asdfa\0sdfasdfg\0we\0yr", 5);
  test_strlen("asdfasdfasdfgweyr1239859102384\0\0", 30);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
