/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <uscl/std/cassert>
#include <uscl/std/initializer_list>
#include <uscl/std/inplace_vector>
#include <uscl/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test_equality()
{
  using inplace_vector = cuda::std::inplace_vector<T, 42ull>;

  inplace_vector vec{T(1), T(42), T(1337), T(0)};
  inplace_vector other_vec{T(0), T(1), T(2), T(3), T(4)};

  auto res_equality = vec == vec;
  static_assert(cuda::std::is_same<decltype(res_equality), bool>::value, "");
  assert(res_equality);

  auto res_inequality = vec != other_vec;
  static_assert(cuda::std::is_same<decltype(res_inequality), bool>::value, "");
  assert(res_inequality);
}

template <class T>
__host__ __device__ constexpr void test_relation()
{
  using inplace_vector = cuda::std::inplace_vector<T, 42ull>;

  inplace_vector vec{T(0), T(1), T(1), T(3), T(4)};
  inplace_vector other_vec{T(0), T(1), T(2), T(3), T(4)};

  auto res_less = vec < other_vec;
  static_assert(cuda::std::is_same<decltype(res_less), bool>::value, "");
  assert(res_less);

  auto res_less_equal = vec <= other_vec;
  static_assert(cuda::std::is_same<decltype(res_less_equal), bool>::value, "");
  assert(res_less_equal);

  auto res_greater = vec > other_vec;
  static_assert(cuda::std::is_same<decltype(res_greater), bool>::value, "");
  assert(!res_greater);

  auto res_greater_equal = vec >= other_vec;
  static_assert(cuda::std::is_same<decltype(res_greater_equal), bool>::value, "");
  assert(!res_greater_equal);
}

template <class T>
__host__ __device__ constexpr void test()
{
  test_equality<T>();
  test_relation<T>();
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<Trivial>();

  if (!cuda::std::is_constant_evaluated())
  {
    test<NonTrivial>();
    test<NonTrivialDestructor>();
    test<ThrowingDefaultConstruct>();
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  return 0;
}
