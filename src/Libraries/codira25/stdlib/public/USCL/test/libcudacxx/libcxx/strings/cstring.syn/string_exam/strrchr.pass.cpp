/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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

#include <uscl/std/__string/constexpr_c_functions.h>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>

template <class T>
__host__ __device__ constexpr void test_strrchr(T* str, T c, T* expected_ret)
{
  const auto ret = cuda::std::__cccl_strrchr(str, c);
  assert(ret == expected_ret);
}

template <class T>
__host__ __device__ constexpr void test_type();

#define TEST_SPECIALIZATION(T, P)                   \
  template <>                                       \
  __host__ __device__ constexpr void test_type<T>() \
  {                                                 \
    {                                               \
      T str[]{P##""};                               \
      test_strrchr<T>(str, P##'\0', str);           \
      test_strrchr<T>(str, P##'a', nullptr);        \
    }                                               \
    {                                               \
      T str[]{P##"a"};                              \
      test_strrchr<T>(str, P##'\0', str + 1);       \
      test_strrchr<T>(str, P##'a', str);            \
      test_strrchr<T>(str, P##'b', nullptr);        \
    }                                               \
    {                                               \
      T str[]{P##"aaa"};                            \
      test_strrchr<T>(str, P##'\0', str + 3);       \
      test_strrchr<T>(str, P##'a', str + 2);        \
      test_strrchr<T>(str, P##'b', nullptr);        \
    }                                               \
    {                                               \
      T str[]{P##"abcdabcd\0\0"};                   \
      test_strrchr<T>(str, P##'\0', str + 8);       \
      test_strrchr<T>(str, P##'a', str + 4);        \
      test_strrchr<T>(str, P##'b', str + 5);        \
      test_strrchr<T>(str, P##'c', str + 6);        \
      test_strrchr<T>(str, P##'d', str + 7);        \
      test_strrchr<T>(str, P##'e', nullptr);        \
      test_strrchr<T>(str, P##'f', nullptr);        \
    }                                               \
  }

TEST_SPECIALIZATION(char, )
#if _CCCL_HAS_CHAR8_T()
TEST_SPECIALIZATION(char8_t, u8)
#endif // _CCCL_HAS_CHAR8_T()
TEST_SPECIALIZATION(char16_t, u)
TEST_SPECIALIZATION(char32_t, U)

__host__ __device__ constexpr bool test()
{
  test_type<char>();
#if _CCCL_HAS_CHAR8_T()
  test_type<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_type<char16_t>();
  test_type<char32_t>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
