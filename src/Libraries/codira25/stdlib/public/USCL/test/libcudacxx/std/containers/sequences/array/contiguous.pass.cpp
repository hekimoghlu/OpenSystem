/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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

// An array is a contiguous container

#include <uscl/std/array>
#include <uscl/std/cassert>
// #include <uscl/std/memory>
#include <uscl/std/utility>

#include "test_macros.h"

template <class Container>
__host__ __device__ constexpr void assert_contiguous(Container const& c)
{
  for (cuda::std::size_t i = 0; i < c.size(); ++i)
  {
    assert(*(c.begin() + i) == *(cuda::std::addressof(*c.begin()) + i));
  }
}

__host__ __device__ constexpr bool tests()
{
  assert_contiguous(cuda::std::array<double, 0>());
  assert_contiguous(cuda::std::array<double, 1>());
  assert_contiguous(cuda::std::array<double, 2>());
  assert_contiguous(cuda::std::array<double, 3>());

  assert_contiguous(cuda::std::array<char, 0>());
  assert_contiguous(cuda::std::array<char, 1>());
  assert_contiguous(cuda::std::array<char, 2>());
  assert_contiguous(cuda::std::array<char, 3>());

  return true;
}

int main(int, char**)
{
  tests();
#if defined(_CCCL_BUILTIN_ADDRESSOF) // begin() & friends are constexpr in >= C++17 only
  static_assert(tests(), "");
#endif // defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
