/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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

// <chrono>
// class day;

// constexpr bool ok() const noexcept;
//  Returns: 1 <= d_ && d_ <= 31

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using day = cuda::std::chrono::day;
  static_assert(noexcept(cuda::std::declval<const day>().ok()));
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::declval<const day>().ok())>);

  static_assert(!day{0}.ok(), "");
  static_assert(day{1}.ok(), "");

  assert(!day{0}.ok());
  for (unsigned i = 1; i <= 31; ++i)
  {
    assert(day{i}.ok());
  }
  for (unsigned i = 32; i <= 255; ++i)
  {
    assert(!day{i}.ok());
  }

  return 0;
}
