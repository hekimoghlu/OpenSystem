/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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

// constexpr byte operator^(byte l, byte r) noexcept;

int main(int, char**)
{
  constexpr cuda::std::byte b1{static_cast<cuda::std::byte>(1)};
  constexpr cuda::std::byte b8{static_cast<cuda::std::byte>(8)};
  constexpr cuda::std::byte b9{static_cast<cuda::std::byte>(9)};

  static_assert(noexcept(b1 ^ b8), "");

  static_assert(cuda::std::to_integer<int>(b1 ^ b8) == 9, "");
  static_assert(cuda::std::to_integer<int>(b1 ^ b9) == 8, "");
  static_assert(cuda::std::to_integer<int>(b8 ^ b9) == 1, "");

  static_assert(cuda::std::to_integer<int>(b8 ^ b1) == 9, "");
  static_assert(cuda::std::to_integer<int>(b9 ^ b1) == 8, "");
  static_assert(cuda::std::to_integer<int>(b9 ^ b8) == 1, "");

  return 0;
}
