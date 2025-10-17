/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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

// constexpr expected& operator=(expected&& rhs) noexcept(see below);
//
// Effects:
// - If this->has_value() && rhs.has_value() is true, no effects.
// - Otherwise, if this->has_value() is true, equivalent to:
//   construct_at(addressof(unex), cuda::std::move(rhs.unex));
//   has_val = false;
// - Otherwise, if rhs.has_value() is true, destroys unex and sets has_val to true.
// - Otherwise, equivalent to unex = cuda::std::move(rhs.error()).
//
// Returns: *this.
//
// Remarks: The exception specification is equivalent to is_nothrow_move_constructible_v<E> &&
// is_nothrow_move_assignable_v<E>.
//
// This operator is defined as deleted unless is_move_constructible_v<E> is true and is_move_assignable_v<E> is true.

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "../../types.h"
#include "test_macros.h"

struct NotMoveConstructible
{
  NotMoveConstructible(NotMoveConstructible&&)            = delete;
  NotMoveConstructible& operator=(NotMoveConstructible&&) = default;
};

struct NotMoveAssignable
{
  NotMoveAssignable(NotMoveAssignable&&)            = default;
  NotMoveAssignable& operator=(NotMoveAssignable&&) = delete;
};

// Test constraints
static_assert(cuda::std::is_move_assignable_v<cuda::std::expected<void, int>>, "");

// !is_move_assignable_v<E>
static_assert(!cuda::std::is_move_assignable_v<cuda::std::expected<void, NotMoveAssignable>>, "");

// !is_move_constructible_v<E>
static_assert(!cuda::std::is_move_assignable_v<cuda::std::expected<void, NotMoveConstructible>>, "");

// Test noexcept
struct MoveCtorMayThrow
{
  __host__ __device__ MoveCtorMayThrow(MoveCtorMayThrow&&) noexcept(false) {}
  MoveCtorMayThrow& operator=(MoveCtorMayThrow&&) noexcept = default;
};

struct MoveAssignMayThrow
{
  MoveAssignMayThrow(MoveAssignMayThrow&&) noexcept = default;
  __host__ __device__ MoveAssignMayThrow& operator=(MoveAssignMayThrow&&) noexcept(false)
  {
    return *this;
  }
};

// Test noexcept
static_assert(cuda::std::is_nothrow_move_assignable_v<cuda::std::expected<void, int>>, "");

// !is_nothrow_move_assignable_v<E>
static_assert(!cuda::std::is_nothrow_move_assignable_v<cuda::std::expected<void, MoveAssignMayThrow>>, "");

// !is_nothrow_move_constructible_v<E>
static_assert(!cuda::std::is_nothrow_move_assignable_v<cuda::std::expected<void, MoveCtorMayThrow>>, "");

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // If this->has_value() && rhs.has_value() is true, no effects.
  {
    cuda::std::expected<void, int> e1;
    cuda::std::expected<void, int> e2;
    decltype(auto) x = (e1 = cuda::std::move(e2));
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<void, int>&>, "");
    assert(&x == &e1);
    assert(e1.has_value());
  }

  // Otherwise, if this->has_value() is true, equivalent to:
  // construct_at(addressof(unex), cuda::std::move(rhs.unex));
  // has_val = false;
  {
    Traced::state state{};
    cuda::std::expected<void, Traced> e1;
    cuda::std::expected<void, Traced> e2(cuda::std::unexpect, state, 5);
    decltype(auto) x = (e1 = cuda::std::move(e2));
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<void, Traced>&>, "");
    assert(&x == &e1);
    assert(!e1.has_value());
    assert(e1.error().data_ == 5);

    assert(state.moveCtorCalled);
  }

  // Otherwise, if rhs.has_value() is true, destroys unex and sets has_val to true.
  {
    Traced::state state{};
    cuda::std::expected<void, Traced> e1(cuda::std::unexpect, state, 5);
    cuda::std::expected<void, Traced> e2;
    decltype(auto) x = (e1 = cuda::std::move(e2));
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<void, Traced>&>, "");
    assert(&x == &e1);
    assert(e1.has_value());

    assert(state.dtorCalled);
  }

  // Otherwise, equivalent to unex = rhs.error().
  {
    Traced::state state{};
    cuda::std::expected<void, Traced> e1(cuda::std::unexpect, state, 5);
    cuda::std::expected<void, Traced> e2(cuda::std::unexpect, state, 10);
    decltype(auto) x = (e1 = cuda::std::move(e2));
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<void, Traced>&>, "");
    assert(&x == &e1);
    assert(!e1.has_value());
    assert(e1.error().data_ == 10);

    assert(state.moveAssignCalled);
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  cuda::std::expected<void, ThrowOnMoveConstruct> e1(cuda::std::in_place);
  cuda::std::expected<void, ThrowOnMoveConstruct> e2(cuda::std::unexpect);
  try
  {
    e1 = cuda::std::move(e2);
    assert(false);
  }
  catch (Except)
  {
    assert(e1.has_value());
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
