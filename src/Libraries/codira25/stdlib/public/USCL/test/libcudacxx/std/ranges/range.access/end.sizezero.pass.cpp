/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc

// cuda::std::ranges::end
// cuda::std::ranges::cend
//   Test the fix for https://llvm.org/PR54100

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_macros.h"

#ifndef __CUDA_ARCH__
struct A
{
  int m[0];
};
static_assert(sizeof(A) == 0); // an extension supported by GCC and Clang

__device__ static A a[10];

int main(int, char**)
{
  auto p = cuda::std::ranges::end(a);
  static_assert(cuda::std::same_as<A*, decltype(cuda::std::ranges::end(a))>);
  assert(p == a + 10);
  auto cp = cuda::std::ranges::cend(a);
  static_assert(cuda::std::same_as<const A*, decltype(cuda::std::ranges::cend(a))>);
  assert(cp == a + 10);

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
