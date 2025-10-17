/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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

// constexpr day operator""d(unsigned long long d) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
#if _LIBCUDACXX_HAS_CXX20_CHRONO_LITERALS()
  {
    using namespace cuda::std::chrono;
    static_assert(noexcept(4d));
    static_assert(cuda::std::is_same_v<day, decltype(4d)>);

    static_assert(7d == day(7));
    day d1 = 4d;
    assert(d1 == day(4));
  }

  {
    using namespace cuda::std::literals;
    static_assert(noexcept(4d));
    static_assert(cuda::std::is_same_v<cuda::std::chrono::day, decltype(4d)>);

    static_assert(7d == cuda::std::chrono::day(7));

    cuda::std::chrono::day d1 = 4d;
    assert(d1 == cuda::std::chrono::day(4));
  }
#endif // _LIBCUDACXX_HAS_CXX20_CHRONO_LITERALS()

  return 0;
}
