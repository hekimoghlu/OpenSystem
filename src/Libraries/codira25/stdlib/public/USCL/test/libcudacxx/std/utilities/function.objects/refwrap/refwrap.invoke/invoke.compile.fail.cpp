/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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

// member data pointer:  cv qualifiers should transfer from argument to return type

struct A_int_1
{
  __host__ __device__ A_int_1()
      : data_(5)
  {}

  int data_;
};

__host__ __device__ void test_int_1()
{
  // member data pointer
  {
    int A_int_1::* fp = &A_int_1::data_;
    cuda::std::reference_wrapper<int A_int_1::*> r1(fp);
    A_int_1 a;
    assert(r1(a) == 5);
    r1(a) = 6;
    assert(r1(a) == 6);
    const A_int_1* ap = &a;
    assert(r1(ap) == 6);
    r1(ap) = 7;
    assert(r1(ap) == 7);
  }
}

int main(int, char**)
{
  test_int_1();

  return 0;
}
