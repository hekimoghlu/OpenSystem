/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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

// ~tuple();

// C++17 added:
//   The destructor of tuple shall be a trivial destructor
//     if (is_trivially_destructible_v<Types> && ...) is true.

#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "test_macros.h"

struct not_trivially_destructible
{
  __host__ __device__ virtual ~not_trivially_destructible() {}
};

int main(int, char**)
{
  static_assert(cuda::std::is_trivially_destructible<cuda::std::tuple<>>::value, "");
  static_assert(cuda::std::is_trivially_destructible<cuda::std::tuple<void*>>::value, "");
  static_assert(cuda::std::is_trivially_destructible<cuda::std::tuple<int, float>>::value, "");
  // cuda::std::string is not supported
  /*
  static_assert(!cuda::std::is_trivially_destructible<
      cuda::std::tuple<not_trivially_destructible> >::value, "");
  static_assert(!cuda::std::is_trivially_destructible<
      cuda::std::tuple<int, not_trivially_destructible> >::value, "");
  */
  // non-string check
  static_assert(!cuda::std::is_trivially_destructible<cuda::std::tuple<not_trivially_destructible>>::value, "");
  static_assert(!cuda::std::is_trivially_destructible<cuda::std::tuple<int, not_trivially_destructible>>::value, "");
  return 0;
}
