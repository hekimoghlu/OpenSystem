/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class U, class... Args>
//   constexpr explicit unexpected(in_place_t, initializer_list<U> il, Args&&... args);
//
// Constraints: is_constructible_v<E, initializer_list<U>&, Args...> is true.
//
// Effects: Direct-non-list-initializes unex with il, cuda::std::forward<Args>(args)....
//
// Throws: Any exception thrown by the initialization of unex.

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/utility>

#include "test_macros.h"

struct Arg
{
  int i;
  __host__ __device__ constexpr Arg(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr Arg(const Arg& other)
      : i(other.i)
  {}
  __host__ __device__ constexpr Arg(Arg&& other)
      : i(other.i)
  {
    other.i = 0;
  }
};

struct Error
{
  cuda::std::initializer_list<int> list;
  Arg arg;
  __host__ __device__ constexpr explicit Error(cuda::std::initializer_list<int> l, const Arg& a)
      : list(l)
      , arg(a)
  {}
  __host__ __device__ constexpr explicit Error(cuda::std::initializer_list<int> l, Arg&& a)
      : list(l)
      , arg(cuda::std::move(a))
  {}
};

// Test Constraints:
static_assert(
  cuda::std::
    constructible_from<cuda::std::unexpected<Error>, cuda::std::in_place_t, cuda::std::initializer_list<int>, Arg>,
  "");

// !is_constructible_v<E, initializer_list<U>&, Args...>
struct Foo
{};
static_assert(
  !cuda::std::
    constructible_from<cuda::std::unexpected<Error>, cuda::std::in_place_t, cuda::std::initializer_list<double>, Arg>,
  "");

// test explicit
template <class T>
__host__ __device__ void conversion_test(T);

template <class T, class... Args>
_CCCL_CONCEPT ImplicitlyConstructible = _CCCL_REQUIRES_EXPR((T, variadic Args), T t, Args&&... args)(
  (conversion_test<T>({cuda::std::forward<Args>(args)...})));

static_assert(ImplicitlyConstructible<int, int>, "");
static_assert(
  !ImplicitlyConstructible<cuda::std::unexpected<Error>, cuda::std::in_place_t, cuda::std::initializer_list<int>, Arg>,
  "");

__host__ __device__ constexpr bool test()
{
  // lvalue
  {
    Arg a{5};
    auto l = {1, 2, 3};
    cuda::std::unexpected<Error> unex(cuda::std::in_place, l, a);
    assert(unex.error().arg.i == 5);

    int expected = 1;
    for (const auto val : unex.error().list)
    {
      assert(val == expected++);
    }
    assert(a.i == 5);
  }

  // rvalue
  {
    Arg a{5};
    auto l = {1, 2, 3};
    cuda::std::unexpected<Error> unex(cuda::std::in_place, l, cuda::std::move(a));
    assert(unex.error().arg.i == 5);

    int expected = 1;
    for (const auto val : unex.error().list)
    {
      assert(val == expected++);
    }
    assert(a.i == 0);
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  struct Except
  {};

  struct Throwing
  {
    Throwing(cuda::std::initializer_list<int>, int)
    {
      throw Except{};
    }
  };

  try
  {
    cuda::std::unexpected<Throwing> u(cuda::std::in_place, {1, 2}, 5);
    assert(false);
  }
  catch (Except)
  {}
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test(), "");
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
