/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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

// template<class U, class G>
//   constexpr explicit(!is_convertible_v<const G&, E>) expected(const expected<U, G>& rhs);
//
// Let GF be const G&
//
// Constraints:
// - is_void_v<U> is true; and
// - is_constructible_v<E, GF> is true; and
// - is_constructible_v<unexpected<E>, expected<U, G>&> is false; and
// - is_constructible_v<unexpected<E>, expected<U, G>> is false; and
// - is_constructible_v<unexpected<E>, const expected<U, G>&> is false; and
// - is_constructible_v<unexpected<E>, const expected<U, G>> is false.
//
// Effects: If rhs.has_value() is false, direct-non-list-initializes unex with cuda::std::forward<GF>(rhs.error()).
//
// Postconditions: rhs.has_value() is unchanged; rhs.has_value() == this->has_value() is true.
//
// Throws: Any exception thrown by the initialization of unex.

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

// Test Constraints:
template <class T1, class Err1, class T2, class Err2>
constexpr bool canCstrFromExpected =
  cuda::std::is_constructible<cuda::std::expected<T1, Err1>, const cuda::std::expected<T2, Err2>&>::value;

struct CtorFromInt
{
  __host__ __device__ CtorFromInt(int);
};

static_assert(canCstrFromExpected<void, CtorFromInt, void, int>, "");

struct NoCtorFromInt
{};

// !is_void_v<E>
static_assert(!canCstrFromExpected<void, int, int, int>, "");

// !is_constructible_v<E, GF>
static_assert(!canCstrFromExpected<void, NoCtorFromInt, void, int>, "");

template <class T>
struct CtorFrom
{
  _CCCL_TEMPLATE(class T2 = T)
  _CCCL_REQUIRES((!cuda::std::same_as<T2, int>) )
  __host__ __device__ explicit CtorFrom(int);
  __host__ __device__ explicit CtorFrom(T);
  template <class U>
  __host__ __device__ explicit CtorFrom(U&&) = delete;
};

// Note for below 4 tests, because their E is constructible from cvref of cuda::std::expected<void, int>,
// unexpected<E> will be constructible from cvref of cuda::std::expected<void, int>
// is_constructible_v<unexpected<E>, expected<U, G>&>
static_assert(!canCstrFromExpected<void, CtorFrom<cuda::std::expected<void, int>&>, void, int>, "");

// is_constructible_v<unexpected<E>, expected<U, G>>
static_assert(!canCstrFromExpected<void, CtorFrom<cuda::std::expected<void, int>&&>, void, int>, "");

// is_constructible_v<unexpected<E>, const expected<U, G>&> is false
static_assert(!canCstrFromExpected<void, CtorFrom<cuda::std::expected<void, int> const&>, void, int>, "");

// is_constructible_v<unexpected<E>, const expected<U, G>>
static_assert(!canCstrFromExpected<void, CtorFrom<cuda::std::expected<void, int> const&&>, void, int>, "");

// test explicit
static_assert(cuda::std::is_convertible_v<const cuda::std::expected<void, int>&, cuda::std::expected<void, long>>, "");

// !is_convertible_v<GF, E>.
static_assert(
  cuda::std::is_constructible_v<cuda::std::expected<void, CtorFrom<int>>, const cuda::std::expected<void, int>&>, "");
static_assert(
  !cuda::std::is_convertible_v<const cuda::std::expected<void, int>&, cuda::std::expected<void, CtorFrom<int>>>, "");

struct Data
{
  int i;
  __host__ __device__ constexpr Data(int ii)
      : i(ii)
  {}
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // convert the error
  {
    const cuda::std::expected<void, int> e1(cuda::std::unexpect, 5);
    cuda::std::expected<void, Data> e2 = e1;
    assert(!e2.has_value());
    assert(e2.error().i == 5);
    assert(!e1.has_value());
    assert(e1.error() == 5);
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  struct Except
  {};

  struct ThrowingInt
  {
    ThrowingInt(int)
    {
      throw Except{};
    }
  };

  // throw on converting error
  {
    const cuda::std::expected<void, int> e1(cuda::std::unexpect);
    try
    {
      cuda::std::expected<void, ThrowingInt> e2 = e1;
      unused(e2);
      assert(false);
    }
    catch (Except)
    {}
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
