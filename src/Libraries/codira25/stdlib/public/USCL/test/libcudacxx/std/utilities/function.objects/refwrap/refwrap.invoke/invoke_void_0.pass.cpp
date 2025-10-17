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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// template <class... ArgTypes>
//   requires Callable<T, ArgTypes&&...>
//   Callable<T, ArgTypes&&...>::result_type
//   operator()(ArgTypes&&... args) const;

// #include <uscl/std/functional>
#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "test_macros.h"

// 0 args, return void

TEST_GLOBAL_VARIABLE int count = 0;

__host__ __device__ void f_void_0()
{
  ++count;
}

struct A_void_0
{
  __host__ __device__ void operator()()
  {
    ++count;
  }
};

__host__ __device__ void test_void_0()
{
  int save_count = count;
  // function
  {
    cuda::std::reference_wrapper<void()> r1(f_void_0);
    r1();
    assert(count == save_count + 1);
    save_count = count;
  }
  // function pointer
  {
    void (*fp)() = f_void_0;
    cuda::std::reference_wrapper<void (*)()> r1(fp);
    r1();
    assert(count == save_count + 1);
    save_count = count;
  }
  // functor
  {
    A_void_0 a0;
    cuda::std::reference_wrapper<A_void_0> r1(a0);
    r1();
    assert(count == save_count + 1);
    save_count = count;
  }
}

int main(int, char**)
{
  test_void_0();

  return 0;
}
