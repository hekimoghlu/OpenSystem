/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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

// UNSUPPORTED: c++17
// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <uscl/std/cassert>
#include <uscl/std/ranges>
#include <uscl/std/tuple>
#include <uscl/std/utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

static_assert(cuda::std::is_convertible_v<ForwardSubrange, cuda::std::pair<ForwardIter, ForwardIter>>);
static_assert(cuda::std::is_convertible_v<ForwardSubrange, cuda::std::tuple<ForwardIter, ForwardIter>>);
static_assert(!cuda::std::is_convertible_v<ForwardSubrange, cuda::std::tuple<ForwardIter, ForwardIter>&>);
static_assert(!cuda::std::is_convertible_v<ForwardSubrange, cuda::std::tuple<ForwardIter, ForwardIter, ForwardIter>>);
static_assert(cuda::std::is_convertible_v<ConvertibleForwardSubrange, cuda::std::tuple<ConvertibleForwardIter, int*>>);
static_assert(!cuda::std::is_convertible_v<SizedIntPtrSubrange, cuda::std::tuple<long*, int*>>);
static_assert(cuda::std::is_convertible_v<SizedIntPtrSubrange, cuda::std::tuple<int*, int*>>);

__host__ __device__ constexpr bool test()
{
  ForwardSubrange a(ForwardIter(globalBuff), ForwardIter(globalBuff + 8));
  cuda::std::pair<ForwardIter, ForwardIter> aPair = a;
  assert(base(aPair.first) == globalBuff);
  assert(base(aPair.second) == globalBuff + 8);
  cuda::std::tuple<ForwardIter, ForwardIter> aTuple = a;
  assert(base(cuda::std::get<0>(aTuple)) == globalBuff);
  assert(base(cuda::std::get<1>(aTuple)) == globalBuff + 8);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
