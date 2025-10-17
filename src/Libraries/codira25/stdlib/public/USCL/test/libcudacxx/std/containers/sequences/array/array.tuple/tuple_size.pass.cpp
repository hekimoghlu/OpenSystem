/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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

// tuple_size<array<T, N> >::value

#include <uscl/std/array>

#include "test_macros.h"

template <class T, cuda::std::size_t N>
__host__ __device__ void test()
{
  {
    typedef cuda::std::array<T, N> C;
    static_assert((cuda::std::tuple_size<C>::value == N), "");
  }
  {
    typedef cuda::std::array<T const, N> C;
    static_assert((cuda::std::tuple_size<C>::value == N), "");
  }
  {
    typedef cuda::std::array<T volatile, N> C;
    static_assert((cuda::std::tuple_size<C>::value == N), "");
  }
  {
    typedef cuda::std::array<T const volatile, N> C;
    static_assert((cuda::std::tuple_size<C>::value == N), "");
  }
}

int main(int, char**)
{
  test<double, 0>();
  test<double, 3>();
  test<double, 5>();

  return 0;
}
