/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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

// UNSUPPORTED: gcc-6

// <memory>

// template <class ForwardIt>
// constexpr void destroy(ForwardIt, ForwardIt);

// #include <uscl/std/memory>
#include <uscl/std/cassert>
#include <uscl/std/memory>
#include <uscl/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

struct Counted
{
  int* counter_ = nullptr;
  __host__ __device__ constexpr Counted(int* counter)
      : counter_(counter)
  {
    ++*counter_;
  }
  __host__ __device__ constexpr Counted(Counted const& other)
      : counter_(other.counter_)
  {
    ++*counter_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~Counted()
  {
    --*counter_;
  }
  __host__ __device__ friend void operator&(Counted) = delete;
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_arrays()
{
  {
    int counter     = 0;
    Counted pool[3] = {{&counter}, {&counter}, {&counter}};
    assert(counter == 3);

    cuda::std::destroy(pool, pool + 3);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::destroy(pool, pool + 3)), void>);
    assert(counter == 0);

    // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
    for (int i = 0; i < 3; ++i)
    {
      cuda::std::__construct_at(pool + i, &counter);
    }
  }
  {
    using Array   = Counted[2];
    int counter   = 0;
    Array pool[3] = {{{&counter}, {&counter}}, {{&counter}, {&counter}}, {{&counter}, {&counter}}};
    assert(counter == 3 * 2);

    cuda::std::destroy(pool, pool + 3);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::destroy(pool, pool + 3)), void>);
    assert(counter == 0);

    // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 2; ++j)
      {
        cuda::std::__construct_at(pool[i] + j, &counter);
      }
    }
  }

  return true;
}

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX20 void test()
{
  int counter     = 0;
  Counted pool[5] = {{&counter}, {&counter}, {&counter}, {&counter}, {&counter}};
  assert(counter == 5);

  cuda::std::destroy(It(pool), It(pool + 5));
  static_assert(cuda::std::is_same_v<decltype(cuda::std::destroy(It(pool), It(pool + 5))), void>);
  assert(counter == 0);

  // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
  for (int i = 0; i < 5; ++i)
  {
    cuda::std::__construct_at(pool + i, &counter);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool tests()
{
  test<Counted*>();
  test<forward_iterator<Counted*>>();
  return true;
}

int main(int, char**)
{
  tests();
  test_arrays();
#if TEST_STD_VER > 2017
#  if !TEST_COMPILER(NVRTC)
#    if TEST_COMPILER(CLANG, >, 10) || TEST_COMPILER(GCC, >, 9) || TEST_COMPILER(MSVC2022) || TEST_COMPILER(NVHPC)
  static_assert(tests());
  // TODO: Until cuda::std::__construct_at has support for arrays, it's impossible to test this
  //       in a constexpr context (see https://reviews.llvm.org/D114903).
  // static_assert(test_arrays());
#    endif
#  endif // TEST_COMPILER(NVRTC)
#endif // TEST_STD_VER > 2017
  return 0;
}
