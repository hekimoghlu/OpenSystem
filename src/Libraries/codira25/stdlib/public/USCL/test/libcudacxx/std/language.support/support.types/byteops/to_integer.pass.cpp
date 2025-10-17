/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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
#include <uscl/std/type_traits>

#include <test_macros.h>

// template <class IntegerType>
//    constexpr IntegerType to_integer(byte b) noexcept;
// This function shall not participate in overload resolution unless
//   is_integral_v<IntegerType> is true.

int main(int, char**)
{
  constexpr cuda::std::byte b1{static_cast<cuda::std::byte>(1)};
  constexpr cuda::std::byte b3{static_cast<cuda::std::byte>(3)};

  static_assert(noexcept(cuda::std::to_integer<int>(b1)), "");
  static_assert(cuda::std::is_same<int, decltype(cuda::std::to_integer<int>(b1))>::value, "");
  static_assert(cuda::std::is_same<long, decltype(cuda::std::to_integer<long>(b1))>::value, "");
  static_assert(cuda::std::is_same<unsigned short, decltype(cuda::std::to_integer<unsigned short>(b1))>::value, "");

  static_assert(cuda::std::to_integer<int>(b1) == 1, "");
  static_assert(cuda::std::to_integer<int>(b3) == 3, "");

  return 0;
}
