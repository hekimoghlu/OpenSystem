/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Types, class Alloc>
//   struct uses_allocator<tuple<Types...>, Alloc> : true_type { };

#include <uscl/std/tuple>
#include <uscl/std/type_traits>

#include "test_macros.h"

struct NonTrivialEmpty
{
  __host__ __device__ NonTrivialEmpty() {}
};
static_assert(cuda::std::is_trivially_copyable<NonTrivialEmpty>::value, "");

struct NonTrivialNonEmpty
{
  int val_ = 0;
  __host__ __device__ NonTrivialNonEmpty() {}
};
static_assert(cuda::std::is_trivially_copyable<NonTrivialNonEmpty>::value, "");

struct NonTriviallyCopyAble
{
  int val_ = 0;
  __host__ __device__ NonTriviallyCopyAble& operator=(const NonTriviallyCopyAble)
  {
    return *this;
  }
};

int main(int, char**)
{
  static_assert(cuda::std::is_trivially_copyable<cuda::std::tuple<int, float>>::value, "");
  static_assert(cuda::std::is_trivially_copyable<cuda::std::tuple<int, NonTrivialEmpty>>::value, "");
  static_assert(cuda::std::is_trivially_copyable<cuda::std::tuple<int, NonTrivialNonEmpty>>::value, "");
  static_assert(!cuda::std::is_trivially_copyable<cuda::std::tuple<int, NonTriviallyCopyAble>>::value, "");

  return 0;
}
