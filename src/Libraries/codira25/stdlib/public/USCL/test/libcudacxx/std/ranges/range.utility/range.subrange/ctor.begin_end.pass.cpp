/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

// convertible-to-non-slicing cases:
//   1. Not convertible (fail)
//   2. Only one is a pointer (succeed)
//   3. Both are not pointers (succeed)
//   4. Pointer elements are different types (fail)
//   5. Pointer elements are same type (succeed)

// !StoreSize ctor.
static_assert(cuda::std::is_constructible_v<ForwardSubrange, ForwardIter, ForwardIter>); // Default case.
static_assert(!cuda::std::is_constructible_v<ForwardSubrange, Empty, ForwardIter>); // 1.
static_assert(cuda::std::is_constructible_v<ConvertibleForwardSubrange, ConvertibleForwardIter, int*>); // 2.
static_assert(cuda::std::is_constructible_v<ForwardSubrange, ForwardIter, ForwardIter>); // 3. (Same as default case.)
// 4. and 5. must be sized.

__host__ __device__ constexpr bool test()
{
  ForwardSubrange a(ForwardIter(globalBuff), ForwardIter(globalBuff + 8));
  assert(base(a.begin()) == globalBuff);
  assert(base(a.end()) == globalBuff + 8);

  ConvertibleForwardSubrange b(ConvertibleForwardIter(globalBuff), globalBuff + 8);
  assert(b.begin() == globalBuff);
  assert(b.end() == globalBuff + 8);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
