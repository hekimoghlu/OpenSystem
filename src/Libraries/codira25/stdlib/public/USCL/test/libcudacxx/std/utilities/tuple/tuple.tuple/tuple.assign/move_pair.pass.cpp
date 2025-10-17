/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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

// template <class U1, class U2>
//   tuple& operator=(pair<U1, U2>&& u);

#include <uscl/std/cassert>
#include <uscl/std/tuple>
#include <uscl/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

struct B
{
  int id_;

  __device__ __host__ explicit B(int i = 0)
      : id_(i)
  {}

  __device__ __host__ virtual ~B() {}
};

struct D : B
{
  __device__ __host__ explicit D(int i)
      : B(i)
  {}
};

int main(int, char**)
{
  {
    using T0 = cuda::std::pair<long, MoveOnly>;
    using T1 = cuda::std::tuple<long long, MoveOnly>;
    T0 t0(2, MoveOnly(3));
    T1 t1;
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1).get() == 3);
  }

  return 0;
}
