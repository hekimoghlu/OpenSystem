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

#include <uscl/std/__string/constexpr_c_functions.h>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>

template <class T>
__host__ __device__ constexpr void test_memcmp(const T* lhs, const T* rhs, size_t n, int expected)
{
  const auto ret = cuda::std::__cccl_memcmp(lhs, rhs, n);

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
  // char
  test_memcmp<char>("abcde", "abcde", 5, 0);
  test_memcmp<char>("abcd1", "abcd0", 5, 1);
  test_memcmp<char>("abcd0", "abcd1", 5, -1);
  test_memcmp<char>("abcd1", "abcd0", 4, 0);
  test_memcmp<char>("abcd0", "abcd1", 4, 0);
  test_memcmp<char>("abcde", "fghij", 5, -1);
  test_memcmp<char>("abcde", "fghij", 0, 0);
  test_memcmp<char>(nullptr, nullptr, 0, 0);

#if _CCCL_HAS_CHAR8_T()
  // char8_t
  test_memcmp<char8_t>(u8"abcde", u8"abcde", 5, 0);
  test_memcmp<char8_t>(u8"abcd1", u8"abcd0", 5, 1);
  test_memcmp<char8_t>(u8"abcd0", u8"abcd1", 5, -1);
  test_memcmp<char8_t>(u8"abcd1", u8"abcd0", 4, 0);
  test_memcmp<char8_t>(u8"abcd0", u8"abcd1", 4, 0);
  test_memcmp<char8_t>(u8"abcde", u8"fghij", 5, -1);
  test_memcmp<char8_t>(u8"abcde", u8"fghij", 0, 0);
  test_memcmp<char8_t>(nullptr, nullptr, 0, 0);
#endif // _CCCL_HAS_CHAR8_T()

  // char16_t
  test_memcmp<char16_t>(u"abcde", u"abcde", 5, 0);
  test_memcmp<char16_t>(u"abcd1", u"abcd0", 5, 1);
  test_memcmp<char16_t>(u"abcd0", u"abcd1", 5, -1);
  test_memcmp<char16_t>(u"abcd1", u"abcd0", 4, 0);
  test_memcmp<char16_t>(u"abcd0", u"abcd1", 4, 0);
  test_memcmp<char16_t>(u"abcde", u"fghij", 5, -1);
  test_memcmp<char16_t>(u"abcde", u"fghij", 0, 0);
  test_memcmp<char16_t>(nullptr, nullptr, 0, 0);

  // char32_t
  test_memcmp<char32_t>(U"abcde", U"abcde", 5, 0);
  test_memcmp<char32_t>(U"abcd1", U"abcd0", 5, 1);
  test_memcmp<char32_t>(U"abcd0", U"abcd1", 5, -1);
  test_memcmp<char32_t>(U"abcd1", U"abcd0", 4, 0);
  test_memcmp<char32_t>(U"abcd0", U"abcd1", 4, 0);
  test_memcmp<char32_t>(U"abcde", U"fghij", 5, -1);
  test_memcmp<char32_t>(U"abcde", U"fghij", 0, 0);
  test_memcmp<char32_t>(nullptr, nullptr, 0, 0);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
