/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#include <uscl/std/type_traits>
#include <uscl/std/utility>

template <class T, cuda::std::size_t... N>
__host__ __device__ constexpr bool equal_buffers(const T* lhs, const T* rhs, cuda::std::index_sequence<N...>)
{
  return ((lhs[N] == rhs[N]) && ...);
}

template <class T, cuda::std::size_t N>
__host__ __device__ constexpr void test_strcpy(const T* str, const T (&ref)[N])
{
  T buff[N]{};
  const auto ret = cuda::std::__cccl_strcpy(buff, str);
  assert(ret == buff);
  assert(equal_buffers(buff, ref, cuda::std::make_index_sequence<N - 1>{}));
}

__host__ __device__ constexpr bool test()
{
  // char
  test_strcpy<char>("", "\0\0\0");
  test_strcpy<char>("a", "a\0\0");
  test_strcpy<char>("a\0", "a\0\0");
  test_strcpy<char>("a\0sdf\0", "a\0\0\0\0\0");
  test_strcpy<char>("hello", "hello\0\0\0\0");
  test_strcpy<char>("hell\0o", "hell\0\0\0");

#if _CCCL_HAS_CHAR8_T()
  // char8_t
  test_strcpy<char8_t>(u8"", u8"\0\0\0");
  test_strcpy<char8_t>(u8"a", u8"a\0\0");
  test_strcpy<char8_t>(u8"a\0", u8"a\0\0");
  test_strcpy<char8_t>(u8"a\0sdf\0", u8"a\0\0\0\0\0");
  test_strcpy<char8_t>(u8"hello", u8"hello\0\0\0\0");
  test_strcpy<char8_t>(u8"hell\0o", u8"hell\0\0\0");
#endif // _CCCL_HAS_CHAR8_T()

  // char16_t
  test_strcpy<char16_t>(u"", u"\0\0\0");
  test_strcpy<char16_t>(u"a", u"a\0\0");
  test_strcpy<char16_t>(u"a\0", u"a\0\0");
  test_strcpy<char16_t>(u"a\0sdf\0", u"a\0\0\0\0\0");
  test_strcpy<char16_t>(u"hello", u"hello\0\0\0\0");
  test_strcpy<char16_t>(u"hell\0o", u"hell\0\0\0");

  // char32_t
  test_strcpy<char32_t>(U"", U"\0\0\0");
  test_strcpy<char32_t>(U"a", U"a\0\0");
  test_strcpy<char32_t>(U"a\0", U"a\0\0");
  test_strcpy<char32_t>(U"a\0sdf\0", U"a\0\0\0\0\0");
  test_strcpy<char32_t>(U"hello", U"hello\0\0\0\0");
  test_strcpy<char32_t>(U"hell\0o", U"hell\0\0\0");

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
