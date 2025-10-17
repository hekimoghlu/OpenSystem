/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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

// XFAIL: gcc-7 && !nvrtc
// GCC 5: Fails for C++11, passes for C++14.
// GCC 6: Fails for C++11, passes for C++14.
// GCC 7: Fails for C++11, fails for C++14.
// GCC 8: Fails for C++11, passes for C++14.

// XFAIL: msvc-19.0

// <cuda/std/functional>

// equal_to, not_equal_to, less, et al.

// Test that these types can be constructed w/o an initializer in a constexpr
// context. This is specifically testing gcc.gnu.org/PR83921

#include <uscl/std/functional>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr bool test_constexpr_context()
{
  [[maybe_unused]] cuda::std::equal_to<T> eq;
  [[maybe_unused]] cuda::std::not_equal_to<T> neq;
  [[maybe_unused]] cuda::std::less<T> l;
  [[maybe_unused]] cuda::std::less_equal<T> le;
  [[maybe_unused]] cuda::std::greater<T> g;
  [[maybe_unused]] cuda::std::greater_equal<T> ge;
  return true;
}

static_assert(test_constexpr_context<int>(), "");
static_assert(test_constexpr_context<void>(), "");

int main(int, char**)
{
  return 0;
}
