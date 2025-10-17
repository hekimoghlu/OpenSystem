/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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

// reference_wrapper(T& t);

// #include <uscl/std/functional>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

class functor1
{};

template <class T>
__host__ __device__ void test(T& t)
{
  cuda::std::reference_wrapper<T> r(t);
  assert(&r.get() == &t);
}

__host__ __device__ void f() {}

int main(int, char**)
{
  void (*fp)() = f;
  test(fp);
  test(f);
  functor1 f1;
  test(f1);
  int i = 0;
  test(i);
  const int j = 0;
  test(j);

  {
    using Ref = cuda::std::reference_wrapper<int>;
    static_assert((cuda::std::is_constructible<Ref, int&>::value), "");
    static_assert((!cuda::std::is_constructible<Ref, int>::value), "");
    static_assert((!cuda::std::is_constructible<Ref, int&&>::value), "");
  }

  {
    using Ref = cuda::std::reference_wrapper<int>;
    static_assert((cuda::std::is_nothrow_constructible<Ref, int&>::value), "");
    static_assert((!cuda::std::is_nothrow_constructible<Ref, int>::value), "");
    static_assert((!cuda::std::is_nothrow_constructible<Ref, int&&>::value), "");
  }

  return 0;
}
