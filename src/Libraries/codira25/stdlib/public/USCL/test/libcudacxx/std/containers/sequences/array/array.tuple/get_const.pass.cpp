/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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

// template <size_t I, class T, size_t N> const T& get(const array<T, N>& a);

#include <uscl/std/array>
#include <uscl/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool tests()
{
  {
    cuda::std::array<double, 1> const array = {3.3};
    assert(cuda::std::get<0>(array) == 3.3);
  }
  {
    cuda::std::array<double, 2> const array = {3.3, 4.4};
    assert(cuda::std::get<0>(array) == 3.3);
    assert(cuda::std::get<1>(array) == 4.4);
  }
  {
    cuda::std::array<double, 3> const array = {3.3, 4.4, 5.5};
    assert(cuda::std::get<0>(array) == 3.3);
    assert(cuda::std::get<1>(array) == 4.4);
    assert(cuda::std::get<2>(array) == 5.5);
  }
  {
    cuda::std::array<double, 1> const array = {3.3};
    static_assert(cuda::std::is_same<double const&, decltype(cuda::std::get<0>(array))>::value, "");
    assert(cuda::std::get<0>(array) == 3.3);
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
