/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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

// <cuda/std/functional>

// template<Returnable R, class T, CopyConstructible... Args>
//   unspecified mem_fn(R (T::* pm)(Args...) const volatile);

#include <uscl/std/cassert>
#include <uscl/std/functional>

struct A
{
  __host__ __device__ char test0() const volatile
  {
    return 'a';
  }
  __host__ __device__ char test1(int) const volatile
  {
    return 'b';
  }
  __host__ __device__ char test2(int, double) const volatile
  {
    return 'c';
  }
};

template <class F>
__host__ __device__ void test0(F f)
{
  {
    A a;
    assert(f(a) == 'a');
    A* ap = &a;
    assert(f(ap) == 'a');
    const volatile A* cap = &a;
    assert(f(cap) == 'a');
    const F& cf = f;
    assert(cf(ap) == 'a');
  }
}

template <class F>
__host__ __device__ void test1(F f)
{
  {
    A a;
    assert(f(a, 1) == 'b');
    A* ap = &a;
    assert(f(ap, 2) == 'b');
    const volatile A* cap = &a;
    assert(f(cap, 2) == 'b');
    const F& cf = f;
    assert(cf(ap, 2) == 'b');
  }
}

template <class F>
__host__ __device__ void test2(F f)
{
  {
    A a;
    assert(f(a, 1, 2) == 'c');
    A* ap = &a;
    assert(f(ap, 2, 3.5) == 'c');
    const volatile A* cap = &a;
    assert(f(cap, 2, 3.5) == 'c');
    const F& cf = f;
    assert(cf(ap, 2, 3.5) == 'c');
  }
}

int main(int, char**)
{
  test0(cuda::std::mem_fn(&A::test0));
  test1(cuda::std::mem_fn(&A::test1));
  test2(cuda::std::mem_fn(&A::test2));

  return 0;
}
