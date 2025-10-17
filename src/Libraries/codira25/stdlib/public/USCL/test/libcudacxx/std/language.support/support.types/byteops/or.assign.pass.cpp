/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
//
//===----------------------------------------------------------------------===//

#include <uscl/std/cstddef>

#include <test_macros.h>

// constexpr byte& operator |=(byte l, byte r) noexcept;

__host__ __device__ constexpr cuda::std::byte test(cuda::std::byte b1, cuda::std::byte b2)
{
  cuda::std::byte bret = b1;
  return bret |= b2;
}

int main(int, char**)
{
  cuda::std::byte b; // not constexpr, just used in noexcept check
  constexpr cuda::std::byte b1{static_cast<cuda::std::byte>(1)};
  constexpr cuda::std::byte b2{static_cast<cuda::std::byte>(2)};
  constexpr cuda::std::byte b8{static_cast<cuda::std::byte>(8)};

  static_assert(noexcept(b |= b), "");

  assert(cuda::std::to_integer<int>(test(b1, b2)) == 3);
  assert(cuda::std::to_integer<int>(test(b1, b8)) == 9);
  assert(cuda::std::to_integer<int>(test(b2, b8)) == 10);

  assert(cuda::std::to_integer<int>(test(b2, b1)) == 3);
  assert(cuda::std::to_integer<int>(test(b8, b1)) == 9);
  assert(cuda::std::to_integer<int>(test(b8, b2)) == 10);

  static_assert(cuda::std::to_integer<int>(test(b1, b2)) == 3, "");
  static_assert(cuda::std::to_integer<int>(test(b1, b8)) == 9, "");
  static_assert(cuda::std::to_integer<int>(test(b2, b8)) == 10, "");

  static_assert(cuda::std::to_integer<int>(test(b2, b1)) == 3, "");
  static_assert(cuda::std::to_integer<int>(test(b8, b1)) == 9, "");
  static_assert(cuda::std::to_integer<int>(test(b8, b2)) == 10, "");

  return 0;
}
