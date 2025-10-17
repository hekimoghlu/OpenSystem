/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

__host__ __device__ constexpr void test_strchr(char* str, int c, char* expected_ret)
{
  const char* cstr = const_cast<const char*>(str);

  // Test cuda::std::strchr(char*, int) overload
  {
    const auto ret = cuda::std::strchr(str, c);
    assert(ret == expected_ret);
  }

  // Test cuda::std::strchr(const char*, int) overload
  {
    const auto ret = cuda::std::strchr(cstr, c);
    assert(ret == expected_ret);
  }
}

__host__ __device__ constexpr bool test()
{
  static_assert(cuda::std::is_same_v<char*, decltype(cuda::std::strchr(cuda::std::declval<char*>(), int{}))>);
  static_assert(
    cuda::std::is_same_v<const char*, decltype(cuda::std::strchr(cuda::std::declval<const char*>(), int{}))>);

  {
    char str[]{""};
    test_strchr(str, '\0', str);
    test_strchr(str, 'a', nullptr);
  }
  {
    char str[]{"a"};
    test_strchr(str, '\0', str + 1);
    test_strchr(str, 'a', str);
    test_strchr(str, 'b', nullptr);
  }
  {
    char str[]{"aaa"};
    test_strchr(str, '\0', str + 3);
    test_strchr(str, 'a', str);
    test_strchr(str, 'b', nullptr);
  }
  {
    char str[]{"abcdabcd"};
    test_strchr(str, '\0', str + 8);
    test_strchr(str, 'a', str);
    test_strchr(str, 'b', str + 1);
    test_strchr(str, 'c', str + 2);
    test_strchr(str, 'd', str + 3);
    test_strchr(str, 'e', nullptr);
    test_strchr(str, 'f', nullptr);
  }

  // Test that searched character is converted from int to char
  {
    char str[]{"a"};
    test_strchr(str, '\0' + cuda::std::numeric_limits<unsigned char>::max() + 1, str + 1);
    test_strchr(str, 'a' + cuda::std::numeric_limits<unsigned char>::max() + 1, str);
    test_strchr(str, 'b' + cuda::std::numeric_limits<unsigned char>::max() + 1, nullptr);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
