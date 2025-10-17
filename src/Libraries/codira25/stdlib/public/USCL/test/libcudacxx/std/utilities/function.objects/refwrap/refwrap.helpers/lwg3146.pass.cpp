/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
//   LWG 3146 "Excessive unwrapping in cuda::std::ref/cref"

// #include <uscl/std/functional>
#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    int i = 0;
    cuda::std::reference_wrapper<int> ri(i);
    cuda::std::reference_wrapper<cuda::std::reference_wrapper<int>> rri(ri);
    auto rrj = cuda::std::ref(rri);
    static_assert(cuda::std::is_same_v<decltype(rrj), cuda::std::reference_wrapper<cuda::std::reference_wrapper<int>>>);
    assert(&rrj.get() == &ri);
  }
  {
    int i = 0;
    cuda::std::reference_wrapper<int> ri(i);
    cuda::std::reference_wrapper<const cuda::std::reference_wrapper<int>> rri(ri);
    auto rrj = cuda::std::ref(rri);
    static_assert(
      cuda::std::is_same_v<decltype(rrj), cuda::std::reference_wrapper<const cuda::std::reference_wrapper<int>>>);
    assert(&rrj.get() == &ri);
  }
  {
    int i = 0;
    cuda::std::reference_wrapper<int> ri(i);
    cuda::std::reference_wrapper<cuda::std::reference_wrapper<int>> rri(ri);
    auto rrj = cuda::std::cref(rri);
    static_assert(
      cuda::std::is_same_v<decltype(rrj), cuda::std::reference_wrapper<const cuda::std::reference_wrapper<int>>>);
    assert(&rrj.get() == &ri);
  }
  {
    int i = 0;
    cuda::std::reference_wrapper<int> ri(i);
    cuda::std::reference_wrapper<const cuda::std::reference_wrapper<int>> rri(ri);
    auto rrj = cuda::std::cref(rri);
    static_assert(
      cuda::std::is_same_v<decltype(rrj), cuda::std::reference_wrapper<const cuda::std::reference_wrapper<int>>>);
    assert(&rrj.get() == &ri);
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && !TEST_COMPILER(NVRTC)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && !TEST_COMPILER(NVRTC)

  return 0;
}
