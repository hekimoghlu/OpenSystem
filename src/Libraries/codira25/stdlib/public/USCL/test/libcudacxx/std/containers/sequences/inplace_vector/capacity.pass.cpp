/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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

#include <uscl/std/__algorithm_>
#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/initializer_list>
#include <uscl/std/inplace_vector>
#include <uscl/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test()
{
  constexpr size_t max_capacity = 42ull;
  using inplace_vector          = cuda::std::inplace_vector<T, max_capacity>;
  inplace_vector range{T(1), T(1337), T(42), T(12), T(0), T(-1)};
  const inplace_vector const_range{T(0), T(42), T(1337), T(42), T(5), T(-42)};

  const auto empty = range.empty();
  static_assert(cuda::std::is_same<decltype(empty), const bool>::value, "");
  assert(!empty);

  const auto const_empty = const_range.empty();
  static_assert(cuda::std::is_same<decltype(const_empty), const bool>::value, "");
  assert(!const_empty);

  const auto size = range.size();
  static_assert(cuda::std::is_same<decltype(size), const typename inplace_vector::size_type>::value, "");
  assert(size == 6);

  const auto const_size = const_range.size();
  static_assert(cuda::std::is_same<decltype(const_size), const typename inplace_vector::size_type>::value, "");
  assert(const_size == 6);

  const auto max_size = range.max_size();
  static_assert(cuda::std::is_same<decltype(max_size), const typename inplace_vector::size_type>::value, "");
  assert(max_size == max_capacity);

  const auto const_max_size = const_range.max_size();
  static_assert(cuda::std::is_same<decltype(const_max_size), const typename inplace_vector::size_type>::value, "");
  assert(const_max_size == max_capacity);

  // max_size is a static member function, so it should also work through the type
  const auto static_max_size = inplace_vector::max_size();
  static_assert(cuda::std::is_same<decltype(static_max_size), const typename inplace_vector::size_type>::value, "");
  assert(static_max_size == max_capacity);

  const auto capacity = range.capacity();
  static_assert(cuda::std::is_same<decltype(capacity), const typename inplace_vector::size_type>::value, "");
  assert(capacity == max_capacity);

  const auto const_capacity = const_range.capacity();
  static_assert(cuda::std::is_same<decltype(const_capacity), const typename inplace_vector::size_type>::value, "");
  assert(const_capacity == max_capacity);

  // capacity is a static member function, so it should also work through the type
  const auto static_capacity = inplace_vector::capacity();
  static_assert(cuda::std::is_same<decltype(static_capacity), const typename inplace_vector::size_type>::value, "");
  assert(static_capacity == max_capacity);
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
