/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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

// reference_wrapper& operator=(const reference_wrapper<T>& x);

// #include <uscl/std/functional>
#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "test_macros.h"

class functor1
{};

struct convertible_to_int_ref
{
  int val = 0;
  __host__ __device__ operator int&()
  {
    return val;
  }
  __host__ __device__ operator int const&() const
  {
    return val;
  }
};

template <class T>
__host__ __device__ void test(T& t)
{
  cuda::std::reference_wrapper<T> r(t);
  T t2 = t;
  cuda::std::reference_wrapper<T> r2(t2);
  r2 = r;
  assert(&r2.get() == &t);
}

__host__ __device__ void f() {}
__host__ __device__ void g() {}

__host__ __device__ void test_function()
{
  cuda::std::reference_wrapper<void()> r(f);
  cuda::std::reference_wrapper<void()> r2(g);
  r2 = r;
  assert(&r2.get() == &f);
}

int main(int, char**)
{
  void (*fp)() = f;
  test(fp);
  test_function();
  functor1 f1;
  test(f1);
  int i = 0;
  test(i);
  const int j = 0;
  test(j);

  convertible_to_int_ref convi{};
  test(convi);
  convertible_to_int_ref const convic{};
  test(convic);

  {
    using Ref = cuda::std::reference_wrapper<int>;
    static_assert((cuda::std::is_assignable<Ref&, int&>::value), "");
    static_assert((!cuda::std::is_assignable<Ref&, int>::value), "");
    static_assert((!cuda::std::is_assignable<Ref&, int&&>::value), "");
  }

  return 0;
}
