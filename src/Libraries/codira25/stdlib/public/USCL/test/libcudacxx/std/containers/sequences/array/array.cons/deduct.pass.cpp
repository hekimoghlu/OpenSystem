/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// UNSUPPORTED: gcc-6, gcc-7
// UNSUPPORTED: clang-9 && nvcc-11.1

// template <class T, class... U>
//   array(T, U...) -> array<T, 1 + sizeof...(U)>;
//
//  Requires: (is_same_v<T, U> && ...) is true. Otherwise the program is ill-formed.

#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>

#include "test_macros.h"

__host__ __device__ constexpr bool tests()
{
  //  Test the explicit deduction guides
  {
    cuda::std::array arr{1, 2, 3}; // array(T, U...)
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<int, 3>>, "");
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
  }

  {
    const long l1 = 42;
    cuda::std::array arr{1L, 4L, 9L, l1}; // array(T, U...)
    static_assert(cuda::std::is_same_v<decltype(arr)::value_type, long>, "");
    static_assert(arr.size() == 4, "");
    assert(arr[0] == 1);
    assert(arr[1] == 4);
    assert(arr[2] == 9);
    assert(arr[3] == l1);
  }

  //  Test the implicit deduction guides
  {
    cuda::std::array<double, 2> source = {4.0, 5.0};
    cuda::std::array arr(source); // array(array)
    static_assert(cuda::std::is_same_v<decltype(arr), decltype(source)>, "");
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<double, 2>>, "");
    assert(arr[0] == 4.0);
    assert(arr[1] == 5.0);
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
