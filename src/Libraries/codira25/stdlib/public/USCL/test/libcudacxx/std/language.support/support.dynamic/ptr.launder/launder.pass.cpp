/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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

// <new>

// template <class T> constexpr T* launder(T* p) noexcept;

#include <uscl/std/__new_>
#include <uscl/std/cassert>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE constexpr int gi   = 5;
TEST_GLOBAL_VARIABLE constexpr float gf = 8.f;

__host__ __device__ constexpr bool test()
{
  assert(cuda::std::launder(&gi) == &gi);
  assert(cuda::std::launder(&gf) == &gf);

  const int* i   = &gi;
  const float* f = &gf;
  static_assert(cuda::std::is_same<decltype(i), decltype(cuda::std::launder(i))>::value, "");
  static_assert(cuda::std::is_same<decltype(f), decltype(cuda::std::launder(f))>::value, "");

  assert(cuda::std::launder(i) == i);
  assert(cuda::std::launder(f) == f);

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_LAUNDER)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_LAUNDER

  return 0;
}
