/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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

template <cuda::std::size_t... N>
__host__ __device__ constexpr bool equal_buffers(const char* lhs, const char* rhs, cuda::std::index_sequence<N...>)
{
  return ((lhs[N] == rhs[N]) && ...);
}

template <cuda::std::size_t N>
__host__ __device__ constexpr void test_strncpy(const char* str, cuda::std::size_t count, const char (&ref)[N])
{
  char buff[N]{};
  for (cuda::std::size_t i = 0; i < N; ++i)
  {
    buff[i] = 'x';
  }

  const auto ret = cuda::std::strncpy(buff, str, count);
  assert(ret == buff);
  assert(equal_buffers(buff, ref, cuda::std::make_index_sequence<N - 1>{}));
}

__host__ __device__ constexpr bool test()
{
  static_assert(
    cuda::std::is_same_v<char*,
                         decltype(cuda::std::strncpy(
                           cuda::std::declval<char*>(), cuda::std::declval<const char*>(), cuda::std::size_t{}))>);

  test_strncpy("", 0, "xxx");
  test_strncpy("", 1, "\0xx");
  test_strncpy("", 2, "\0\0x");
  test_strncpy("", 3, "\0\0\0");

  test_strncpy("a", 0, "xxx");
  test_strncpy("a", 1, "axx");
  test_strncpy("a", 2, "a\0x");
  test_strncpy("a", 3, "a\0\0");

  test_strncpy("\0a", 0, "xxx");
  test_strncpy("\0a", 1, "\0xx");
  test_strncpy("\0a", 2, "\0\0x");
  test_strncpy("\0a", 3, "\0\0\0");

  test_strncpy("hello", 5, "helloxxx");
  test_strncpy("hello", 6, "hello\0xx");
  test_strncpy("hello", 7, "hello\0\0x");
  test_strncpy("hello", 8, "hello\0\0\0");

  test_strncpy("hell\0o", 4, "hellxxxx");
  test_strncpy("hell\0o", 5, "hell\0xx");
  test_strncpy("hell\0o", 6, "hell\0\0x");
  test_strncpy("hell\0o", 7, "hell\0\0\0");

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
