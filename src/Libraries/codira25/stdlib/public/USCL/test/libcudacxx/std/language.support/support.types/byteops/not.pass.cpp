/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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

// constexpr byte operator~(byte b) noexcept;

int main(int, char**)
{
  constexpr cuda::std::byte b1{static_cast<cuda::std::byte>(1)};
  constexpr cuda::std::byte b2{static_cast<cuda::std::byte>(2)};
  constexpr cuda::std::byte b8{static_cast<cuda::std::byte>(8)};

  static_assert(noexcept(~b1), "");

  static_assert(cuda::std::to_integer<int>(~b1) == 254, "");
  static_assert(cuda::std::to_integer<int>(~b2) == 253, "");
  static_assert(cuda::std::to_integer<int>(~b8) == 247, "");

  return 0;
}
