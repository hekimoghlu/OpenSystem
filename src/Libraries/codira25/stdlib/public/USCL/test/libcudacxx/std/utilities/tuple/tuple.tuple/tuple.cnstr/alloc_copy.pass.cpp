/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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

// template <class Alloc>
//   tuple(allocator_arg_t, const Alloc& a, const tuple&);

#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "../alloc_first.h"
#include "../alloc_last.h"
#include "allocators.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    using T = cuda::std::tuple<>;
    T t0;
    T t(cuda::std::allocator_arg, A1<int>(), t0);
  }
  {
    using T = cuda::std::tuple<int>;
    T t0(2);
    T t(cuda::std::allocator_arg, A1<int>(), t0);
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    using T = cuda::std::tuple<alloc_first>;
    T t0(2);
    alloc_first::allocator_constructed() = false;
    T t(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    using T = cuda::std::tuple<alloc_last>;
    T t0(2);
    alloc_last::allocator_constructed() = false;
    T t(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == 2);
  }
// testing extensions
#ifdef _LIBCUDACXX_VERSION
  {
    using T = cuda::std::tuple<alloc_first, alloc_last>;
    T t0(2, 3);
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    T t(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == 2);
    assert(cuda::std::get<1>(t) == 3);
  }
  {
    using T = cuda::std::tuple<int, alloc_first, alloc_last>;
    T t0(1, 2, 3);
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    T t(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t) == 2);
    assert(cuda::std::get<2>(t) == 3);
  }
#endif

  return 0;
}
