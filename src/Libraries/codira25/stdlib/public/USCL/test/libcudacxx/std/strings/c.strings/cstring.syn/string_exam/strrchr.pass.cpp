/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#include <uscl/std/limits>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr void test_strrchr(char* str, int c, char* expected_ret)
{
  const char* cstr = const_cast<const char*>(str);

  // Test cuda::std::strrchr(char*, int) overload
  {
    const auto ret = cuda::std::strrchr(str, c);
    assert(ret == expected_ret);
  }

  // Test cuda::std::strrchr(const char*, int) overload
  {
    const auto ret = cuda::std::strrchr(cstr, c);
    assert(ret == expected_ret);
  }
}

__host__ __device__ constexpr bool test()
{
  static_assert(cuda::std::is_same_v<char*, decltype(cuda::std::strrchr(cuda::std::declval<char*>(), int{}))>);
  static_assert(
    cuda::std::is_same_v<const char*, decltype(cuda::std::strrchr(cuda::std::declval<const char*>(), int{}))>);

  {
    char str[]{""};
    test_strrchr(str, '\0', str);
    test_strrchr(str, 'a', nullptr);
  }
  {
    char str[]{"a"};
    test_strrchr(str, '\0', str + 1);
    test_strrchr(str, 'a', str);
    test_strrchr(str, 'b', nullptr);
  }
  {
    char str[]{"aaa"};
    test_strrchr(str, '\0', str + 3);
    test_strrchr(str, 'a', str + 2);
    test_strrchr(str, 'b', nullptr);
  }
  {
    char str[]{"abcdabcd\0\0"};
    test_strrchr(str, '\0', str + 8);
    test_strrchr(str, 'a', str + 4);
    test_strrchr(str, 'b', str + 5);
    test_strrchr(str, 'c', str + 6);
    test_strrchr(str, 'd', str + 7);
    test_strrchr(str, 'e', nullptr);
    test_strrchr(str, 'f', nullptr);
  }

  // Test that searched character is converted from int to char
  {
    char str[]{"a"};
    test_strrchr(str, '\0' + cuda::std::numeric_limits<unsigned char>::max() + 1, str + 1);
    test_strrchr(str, 'a' + cuda::std::numeric_limits<unsigned char>::max() + 1, str);
    test_strrchr(str, 'b' + cuda::std::numeric_limits<unsigned char>::max() + 1, nullptr);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
