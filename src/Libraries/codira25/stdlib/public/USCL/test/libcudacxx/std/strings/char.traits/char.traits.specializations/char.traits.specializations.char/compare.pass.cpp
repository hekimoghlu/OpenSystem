/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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

#include <uscl/std/__string_>
#include <uscl/std/cassert>

__host__ __device__ constexpr bool test()
{
  assert(cuda::std::char_traits<char>::compare("", "", 0) == 0);
  assert(cuda::std::char_traits<char>::compare(nullptr, nullptr, 0) == 0);

  assert(cuda::std::char_traits<char>::compare("1", "1", 1) == 0);
  assert(cuda::std::char_traits<char>::compare("1", "2", 1) < 0);
  assert(cuda::std::char_traits<char>::compare("2", "1", 1) > 0);

  assert(cuda::std::char_traits<char>::compare("12", "12", 2) == 0);
  assert(cuda::std::char_traits<char>::compare("12", "13", 2) < 0);
  assert(cuda::std::char_traits<char>::compare("12", "22", 2) < 0);
  assert(cuda::std::char_traits<char>::compare("13", "12", 2) > 0);
  assert(cuda::std::char_traits<char>::compare("22", "12", 2) > 0);

  assert(cuda::std::char_traits<char>::compare("123", "123", 3) == 0);
  assert(cuda::std::char_traits<char>::compare("123", "223", 3) < 0);
  assert(cuda::std::char_traits<char>::compare("123", "133", 3) < 0);
  assert(cuda::std::char_traits<char>::compare("123", "124", 3) < 0);
  assert(cuda::std::char_traits<char>::compare("223", "123", 3) > 0);
  assert(cuda::std::char_traits<char>::compare("133", "123", 3) > 0);
  assert(cuda::std::char_traits<char>::compare("124", "123", 3) > 0);

  {
    char a[] = {static_cast<char>(-1), 0};
    char b[] = {1, 0};
    assert(cuda::std::char_traits<char>::compare(a, b, 1) > 0);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
