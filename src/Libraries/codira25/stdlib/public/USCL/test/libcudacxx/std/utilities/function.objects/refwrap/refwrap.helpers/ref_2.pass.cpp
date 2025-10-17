/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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

// template <ObjectType T> reference_wrapper<T> ref(reference_wrapper<T> t);

// #include <uscl/std/functional>
#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "counting_predicates.h"
#include "test_macros.h"

__host__ __device__ bool is5(int i)
{
  return i == 5;
}

template <typename T>
__host__ __device__ bool call_pred(T pred)
{
  return pred(5);
}

namespace adl
{
struct A
{};
__host__ __device__ void ref(A) {}
} // namespace adl

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    int i                                = 0;
    cuda::std::reference_wrapper<int> r1 = cuda::std::ref(i);
    cuda::std::reference_wrapper<int> r2 = cuda::std::ref(r1);
    assert(&r2.get() == &i);
  }
  {
    adl::A a{};
    cuda::std::reference_wrapper<adl::A> a1 = cuda::std::ref(a);
    cuda::std::reference_wrapper<adl::A> a2 = cuda::std::ref(a1);
    assert(&a2.get() == &a);
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && !TEST_COMPILER(NVRTC)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && !TEST_COMPILER(NVRTC)

  {
    unary_counting_predicate<bool (*)(int), int> cp(is5);
    assert(!cp(6));
    assert(cp.count() == 1);
    assert(call_pred(cp));
    assert(cp.count() == 1);
    assert(call_pred(cuda::std::ref(cp)));
    assert(cp.count() == 2);
  }

  return 0;
}
