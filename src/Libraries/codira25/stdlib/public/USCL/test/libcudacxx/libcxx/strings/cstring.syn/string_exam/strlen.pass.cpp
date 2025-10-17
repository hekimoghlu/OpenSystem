/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
__host__ __device__ constexpr void test_strlen(const T* str, cuda::std::size_t expected)
{
  const auto ret = cuda::std::__cccl_strlen(str);
  assert(ret == expected);
}

__host__ __device__ constexpr bool test()
{
  // char
  test_strlen<char>("", 0);
  test_strlen<char>("a", 1);
  test_strlen<char>("hi", 2);
  test_strlen<char>("asdfgh", 6);
  test_strlen<char>("asdfasdfasdfgweyr", 17);
  test_strlen<char>("asdfasdfasdfgweyr1239859102384", 30);
  test_strlen<char>("\0\0", 0);
  test_strlen<char>("a\0", 1);
  test_strlen<char>("h\0i", 1);
  test_strlen<char>("asdf\0g\0h", 4);
  test_strlen<char>("asdfa\0sdfasdfg\0we\0yr", 5);
  test_strlen<char>("asdfasdfasdfgweyr1239859102384\0\0", 30);

#if _CCCL_HAS_CHAR8_T()
  // char8_t
  test_strlen<char8_t>(u8"", 0);
  test_strlen<char8_t>(u8"a", 1);
  test_strlen<char8_t>(u8"hi", 2);
  test_strlen<char8_t>(u8"asdfgh", 6);
  test_strlen<char8_t>(u8"asdfasdfasdfgweyr", 17);
  test_strlen<char8_t>(u8"asdfasdfasdfgweyr1239859102384", 30);
  test_strlen<char8_t>(u8"\0\0", 0);
  test_strlen<char8_t>(u8"a\0", 1);
  test_strlen<char8_t>(u8"h\0i", 1);
  test_strlen<char8_t>(u8"asdf\0g\0h", 4);
  test_strlen<char8_t>(u8"asdfa\0sdfasdfg\0we\0yr", 5);
  test_strlen<char8_t>(u8"asdfasdfasdfgweyr1239859102384\0\0", 30);
#endif // _CCCL_HAS_CHAR8_T()

  // char16_t
  test_strlen<char16_t>(u"", 0);
  test_strlen<char16_t>(u"a", 1);
  test_strlen<char16_t>(u"hi", 2);
  test_strlen<char16_t>(u"asdfgh", 6);
  test_strlen<char16_t>(u"asdfasdfasdfgweyr", 17);
  test_strlen<char16_t>(u"asdfasdfasdfgweyr1239859102384", 30);
  test_strlen<char16_t>(u"\0\0", 0);
  test_strlen<char16_t>(u"a\0", 1);
  test_strlen<char16_t>(u"h\0i", 1);
  test_strlen<char16_t>(u"asdf\0g\0h", 4);
  test_strlen<char16_t>(u"asdfa\0sdfasdfg\0we\0yr", 5);
  test_strlen<char16_t>(u"asdfasdfasdfgweyr1239859102384\0\0", 30);

  // char32_t
  test_strlen<char32_t>(U"", 0);
  test_strlen<char32_t>(U"a", 1);
  test_strlen<char32_t>(U"hi", 2);
  test_strlen<char32_t>(U"asdfgh", 6);
  test_strlen<char32_t>(U"asdfasdfasdfgweyr", 17);
  test_strlen<char32_t>(U"asdfasdfasdfgweyr1239859102384", 30);
  test_strlen<char32_t>(U"\0\0", 0);
  test_strlen<char32_t>(U"a\0", 1);
  test_strlen<char32_t>(U"h\0i", 1);
  test_strlen<char32_t>(U"asdf\0g\0h", 4);
  test_strlen<char32_t>(U"asdfa\0sdfasdfg\0we\0yr", 5);
  test_strlen<char32_t>(U"asdfasdfasdfgweyr1239859102384\0\0", 30);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
