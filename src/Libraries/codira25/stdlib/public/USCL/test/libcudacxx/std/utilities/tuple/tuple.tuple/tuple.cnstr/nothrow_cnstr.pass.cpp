/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

// tuple(tuple&& u);

#include <uscl/std/cassert>
#include <uscl/std/tuple>
#include <uscl/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

struct NothrowConstruct
{
  __host__ __device__ constexpr NothrowConstruct(int) noexcept {};
};

int main(int, char**)
{
  {
    using T = cuda::std::tuple<NothrowConstruct, NothrowConstruct>;
    T t(0, 1);
    unused(t); // Prevent unused warning

    // Test that tuple<> handles noexcept properly
    static_assert(cuda::std::is_nothrow_constructible<T, int, int>(), "");
    static_assert(cuda::std::is_nothrow_constructible<NothrowConstruct, int>(), "");
  }

  return 0;
}
