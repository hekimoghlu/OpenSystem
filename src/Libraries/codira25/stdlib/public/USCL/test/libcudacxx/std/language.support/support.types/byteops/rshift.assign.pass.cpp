/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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

// template <class IntegerType>
//   constexpr byte& operator>>=(byte& b, IntegerType shift) noexcept;
// This function shall not participate in overload resolution unless
//   is_integral_v<IntegerType> is true.

__host__ __device__ constexpr cuda::std::byte test(cuda::std::byte b)
{
  return b >>= 2;
}

int main(int, char**)
{
  cuda::std::byte b; // not constexpr, just used in noexcept check
  constexpr cuda::std::byte b16{static_cast<cuda::std::byte>(16)};
  constexpr cuda::std::byte b192{static_cast<cuda::std::byte>(192)};

  static_assert(noexcept(b >>= 2), "");

  assert(cuda::std::to_integer<int>(test(b16)) == 4);
  assert(cuda::std::to_integer<int>(test(b192)) == 48);

  static_assert(cuda::std::to_integer<int>(test(b16)) == 4, "");
  static_assert(cuda::std::to_integer<int>(test(b192)) == 48, "");

  return 0;
}
