/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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

// Older Clangs do not support the C++20 feature to constrain destructors

// constexpr void swap(expected& rhs) noexcept(see below);
//
// Constraints:
// is_swappable_v<E> is true and is_move_constructible_v<E> is true.
//
// Throws: Any exception thrown by the expressions in the Effects.
//
// Remarks: The exception specification is equivalent to:
// is_nothrow_move_constructible_v<E> && is_nothrow_swappable_v<E>.

#include <uscl/std/cassert>
#include <uscl/std/expected>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "../../types.h"
#include "test_macros.h"

// Test Constraints:
template <class E>
_CCCL_CONCEPT HasMemberSwap =
  _CCCL_REQUIRES_EXPR((E), cuda::std::expected<void, E> x, cuda::std::expected<void, E> y)((x.swap(y)));

static_assert(HasMemberSwap<int>, "");

struct NotSwappable
{};
__host__ __device__ void swap(NotSwappable&, NotSwappable&) = delete;

// !is_swappable_v<E>
static_assert(!HasMemberSwap<NotSwappable>, "");

struct NotMoveContructible
{
  NotMoveContructible(NotMoveContructible&&) = delete;
  __host__ __device__ friend void swap(NotMoveContructible&, NotMoveContructible&) {}
};

// !is_move_constructible_v<E>
static_assert(!HasMemberSwap<NotMoveContructible>, "");

// Test noexcept
struct MoveMayThrow
{
  __host__ __device__ MoveMayThrow(MoveMayThrow&&) noexcept(false);
  __host__ __device__ friend void swap(MoveMayThrow&, MoveMayThrow&) noexcept {}
};

template <class E, bool = HasMemberSwap<E>>
constexpr bool MemberSwapNoexcept = false;

template <class E>
constexpr bool MemberSwapNoexcept<E, true> = noexcept(
  cuda::std::declval<cuda::std::expected<void, E>&>().swap(cuda::std::declval<cuda::std::expected<void, E>&>()));

static_assert(MemberSwapNoexcept<int>, "");

// !is_nothrow_move_constructible_v<E>
static_assert(!MemberSwapNoexcept<MoveMayThrow>, "");

struct SwapMayThrow
{
  __host__ __device__ friend void swap(SwapMayThrow&, SwapMayThrow&) noexcept(false) {}
};

// !is_nothrow_swappable_v<E>
static_assert(!MemberSwapNoexcept<SwapMayThrow>, "");

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // this->has_value() && rhs.has_value()
  {
    cuda::std::expected<void, int> x;
    cuda::std::expected<void, int> y;
    x.swap(y);

    assert(x.has_value());
    assert(y.has_value());
  }

  // !this->has_value() && !rhs.has_value()
  {
    cuda::std::expected<void, ADLSwap> x(cuda::std::unexpect, 5);
    cuda::std::expected<void, ADLSwap> y(cuda::std::unexpect, 10);
    x.swap(y);

    assert(!x.has_value());
    assert(x.error().i == 10);
    assert(x.error().adlSwapCalled);
    assert(!y.has_value());
    assert(y.error().i == 5);
    assert(y.error().adlSwapCalled);
  }

  // this->has_value() && !rhs.has_value()
  {
    Traced::state s{};
    cuda::std::expected<void, Traced> e1(cuda::std::in_place);
    cuda::std::expected<void, Traced> e2(cuda::std::unexpect, s, 10);

    e1.swap(e2);

    assert(!e1.has_value());
    assert(e1.error().data_ == 10);
    assert(e2.has_value());

    assert(s.moveCtorCalled);
    assert(s.dtorCalled);
  }

  // !this->has_value() && rhs.has_value()
  {
    Traced::state s{};
    cuda::std::expected<void, Traced> e1(cuda::std::unexpect, s, 10);
    cuda::std::expected<void, Traced> e2(cuda::std::in_place);

    e1.swap(e2);

    assert(e1.has_value());
    assert(!e2.has_value());
    assert(e2.error().data_ == 10);

    assert(s.moveCtorCalled);
    assert(s.dtorCalled);
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  // !e1.has_value() && e2.has_value()
  {
    bool e1Destroyed = false;
    cuda::std::expected<void, ThrowOnMove> e1(cuda::std::unexpect, e1Destroyed);
    cuda::std::expected<void, ThrowOnMove> e2(cuda::std::in_place);
    try
    {
      e1.swap(e2);
      assert(false);
    }
    catch (Except)
    {
      assert(!e1.has_value());
      assert(e2.has_value());
      assert(!e1Destroyed);
    }
  }

  // e1.has_value() && !e2.has_value()
  {
    bool e2Destroyed = false;
    cuda::std::expected<void, ThrowOnMove> e1(cuda::std::in_place);
    cuda::std::expected<void, ThrowOnMove> e2(cuda::std::unexpect, e2Destroyed);
    try
    {
      e1.swap(e2);
      assert(false);
    }
    catch (Except)
    {
      assert(e1.has_value());
      assert(!e2.has_value());
      assert(!e2Destroyed);
    }
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
