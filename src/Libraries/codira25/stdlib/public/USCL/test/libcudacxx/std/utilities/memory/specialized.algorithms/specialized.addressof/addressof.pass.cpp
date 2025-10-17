/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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

// <cuda/std/memory>

// template <ObjectType T> T* addressof(T& r);

#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

#if TEST_CUDA_COMPILER(CLANG)
#  include <new>
#endif // TEST_CUDA_COMPILER(CLANG)

struct A
{
  __host__ __device__ void operator&() const {}
};

struct nothing
{
  __host__ __device__ operator char&()
  {
    static char c;
    return c;
  }
};

int main(int, char**)
{
  {
    int i;
    double d;
    assert(cuda::std::addressof(i) == &i);
    assert(cuda::std::addressof(d) == &d);
    A* tp        = new A;
    const A* ctp = tp;
    assert(cuda::std::addressof(*tp) == tp);
    assert(cuda::std::addressof(*ctp) == tp);
    delete tp;
  }
  {
    union
    {
      nothing n;
      int i;
    };
    assert(cuda::std::addressof(n) == (void*) cuda::std::addressof(i));
  }

  return 0;
}
