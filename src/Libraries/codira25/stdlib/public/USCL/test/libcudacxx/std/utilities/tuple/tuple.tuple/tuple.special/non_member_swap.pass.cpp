/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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

// template <class... Types>
//   void swap(tuple<Types...>& x, tuple<Types...>& y);

#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "MoveOnly.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    using T = cuda::std::tuple<>;
    T t0;
    T t1;
    swap(t0, t1);
  }
  {
    using T = cuda::std::tuple<MoveOnly>;
    T t0(MoveOnly(0));
    T t1(MoveOnly(1));
    swap(t0, t1);
    assert(cuda::std::get<0>(t0) == 1);
    assert(cuda::std::get<0>(t1) == 0);
  }
  {
    using T = cuda::std::tuple<MoveOnly, MoveOnly>;
    T t0(MoveOnly(0), MoveOnly(1));
    T t1(MoveOnly(2), MoveOnly(3));
    swap(t0, t1);
    assert(cuda::std::get<0>(t0) == 2);
    assert(cuda::std::get<1>(t0) == 3);
    assert(cuda::std::get<0>(t1) == 0);
    assert(cuda::std::get<1>(t1) == 1);
  }
  {
    using T = cuda::std::tuple<MoveOnly, MoveOnly, MoveOnly>;
    T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
    T t1(MoveOnly(3), MoveOnly(4), MoveOnly(5));
    swap(t0, t1);
    assert(cuda::std::get<0>(t0) == 3);
    assert(cuda::std::get<1>(t0) == 4);
    assert(cuda::std::get<2>(t0) == 5);
    assert(cuda::std::get<0>(t1) == 0);
    assert(cuda::std::get<1>(t1) == 1);
    assert(cuda::std::get<2>(t1) == 2);
  }
  return 0;
}
