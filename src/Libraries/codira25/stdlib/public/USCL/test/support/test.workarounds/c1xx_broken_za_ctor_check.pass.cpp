/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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

// Verify TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK.

#include <uscl/std/type_traits>

#include "test_macros.h"
#include "test_workarounds.h"

struct X
{
  __host__ __device__ X(int) {}

  X(X&&)            = default;
  X& operator=(X&&) = default;

private:
  X(const X&)            = default;
  X& operator=(const X&) = default;
};

__host__ __device__ void PushFront(X&&) {}

template <class T = int>
__host__ __device__ auto test(int) -> decltype(PushFront(cuda::std::declval<T>()), cuda::std::true_type{});
__host__ __device__ auto test(long) -> cuda::std::false_type;

int main(int, char**)
{
#if defined(TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK)
  static_assert(!decltype(test(0))::value, "");
#else
  static_assert(decltype(test(0))::value, "");
#endif

  return 0;
}
