/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#if _CCCL_HAS_CHAR8_T()
  assert(cuda::std::char_traits<char8_t>::compare(u8"", u8"", 0) == 0);
  assert(cuda::std::char_traits<char8_t>::compare(nullptr, nullptr, 0) == 0);

  assert(cuda::std::char_traits<char8_t>::compare(u8"1", u8"1", 1) == 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"1", u8"2", 1) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"2", u8"1", 1) > 0);

  assert(cuda::std::char_traits<char8_t>::compare(u8"12", u8"12", 2) == 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"12", u8"13", 2) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"12", u8"22", 2) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"13", u8"12", 2) > 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"22", u8"12", 2) > 0);

  assert(cuda::std::char_traits<char8_t>::compare(u8"123", u8"123", 3) == 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"123", u8"223", 3) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"123", u8"133", 3) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"123", u8"124", 3) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"223", u8"123", 3) > 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"133", u8"123", 3) > 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"124", u8"123", 3) > 0);
#endif // _CCCL_HAS_CHAR8_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
