/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr sentinel(sentinel<!Const> s)
//   requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "../types.h"

struct Sent
{
  int i = 0;
  __host__ __device__ constexpr Sent() noexcept {}
  __host__ __device__ constexpr Sent(int ii)
      : i(ii)
  {}
  __host__ __device__ friend constexpr bool operator==(int* iter, const Sent& s)
  {
    return s.i > *iter;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator==(const Sent& s, int* iter)
  {
    return s.i > *iter;
  }
  __host__ __device__ friend constexpr bool operator!=(int* iter, const Sent& s)
  {
    return s.i <= *iter;
  }
  __host__ __device__ friend constexpr bool operator!=(const Sent& s, int* iter)
  {
    return s.i <= *iter;
  }
#endif // TEST_STD_VER <= 2017
};

struct ConstSent
{
  int i = 0;
  __host__ __device__ constexpr ConstSent() noexcept {}
  __host__ __device__ constexpr ConstSent(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr ConstSent(const Sent& s)
      : i(s.i)
  {}
  __host__ __device__ friend constexpr bool operator==(int* iter, const ConstSent& s)
  {
    return s.i > *iter;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator==(const ConstSent& s, int* iter)
  {
    return s.i > *iter;
  }
  __host__ __device__ friend constexpr bool operator!=(int* iter, const ConstSent& s)
  {
    return s.i <= *iter;
  }
  __host__ __device__ friend constexpr bool operator!=(const ConstSent& s, int* iter)
  {
    return s.i <= *iter;
  }
#endif // TEST_STD_VER <= 2017
};

struct Range : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const
  {
    return nullptr;
  }
  __host__ __device__ Sent end()
  {
    return Sent{};
  }
  __host__ __device__ ConstSent end() const
  {
    return ConstSent{};
  }
};

struct Pred
{
  __host__ __device__ bool operator()(int i) const;
};

struct NonConvertConstSent
{
  int i = 0;
  __host__ __device__ constexpr NonConvertConstSent() noexcept {}
  __host__ __device__ constexpr NonConvertConstSent(int ii)
      : i(ii)
  {}
  __host__ __device__ friend constexpr bool operator==(int* iter, const NonConvertConstSent& s)
  {
    return s.i > *iter;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator==(const NonConvertConstSent& s, int* iter)
  {
    return s.i > *iter;
  }
  __host__ __device__ friend constexpr bool operator!=(int* iter, const NonConvertConstSent& s)
  {
    return s.i <= *iter;
  }
  __host__ __device__ friend constexpr bool operator!=(const NonConvertConstSent& s, int* iter)
  {
    return s.i <= *iter;
  }
#endif // TEST_STD_VER <= 2017
};

struct NonConvertConstSentRange : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const
  {
    return nullptr;
  };
  __host__ __device__ Sent end()
  {
    return Sent{};
  }
  __host__ __device__ NonConvertConstSent end() const
  {
    return NonConvertConstSent{};
  }
};

// Test Constraint
static_assert(
  cuda::std::is_constructible_v<cuda::std::ranges::sentinel_t<const cuda::std::ranges::take_while_view<Range, Pred>>,
                                cuda::std::ranges::sentinel_t<cuda::std::ranges::take_while_view<Range, Pred>>>);

// !Const
static_assert(
  !cuda::std::is_constructible_v<cuda::std::ranges::sentinel_t<cuda::std::ranges::take_while_view<Range, Pred>>,
                                 cuda::std::ranges::sentinel_t<const cuda::std::ranges::take_while_view<Range, Pred>>>);

// !convertible_to<sentinel_t<V>, sentinel_t<Base>>
static_assert(!cuda::std::is_constructible_v<
              cuda::std::ranges::sentinel_t<const cuda::std::ranges::take_while_view<NonConvertConstSentRange, Pred>>,
              cuda::std::ranges::sentinel_t<cuda::std::ranges::take_while_view<NonConvertConstSentRange, Pred>>>);

struct MoveOnlyConvert
{
  int i = 0;
  __host__ __device__ constexpr MoveOnlyConvert() noexcept {}
  __host__ __device__ constexpr MoveOnlyConvert(Sent&& s)
      : i(s.i)
  {
    s.i = 0;
  }
  __host__ __device__ constexpr friend bool operator==(const MoveOnlyConvert& s, int* iter)
  {
    return s.i > *iter;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ constexpr friend bool operator==(int* iter, const MoveOnlyConvert& s)
  {
    return s.i > *iter;
  }
  __host__ __device__ constexpr friend bool operator!=(const MoveOnlyConvert& s, int* iter)
  {
    return s.i <= *iter;
  }
  __host__ __device__ constexpr friend bool operator!=(int* iter, const MoveOnlyConvert& s)
  {
    return s.i <= *iter;
  }
#endif // TEST_STD_VER <= 2017
};

struct Rng : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const
  {
    return nullptr;
  };
  __host__ __device__ Sent end()
  {
    return Sent{};
  }
  __host__ __device__ MoveOnlyConvert end() const
  {
    return MoveOnlyConvert{};
  }
};

__host__ __device__ constexpr bool test()
{
  // base is init correctly
  {
    using R             = cuda::std::ranges::take_while_view<Range, bool (*)(int)>;
    using Sentinel      = cuda::std::ranges::sentinel_t<R>;
    using ConstSentinel = cuda::std::ranges::sentinel_t<const R>;
    static_assert(!cuda::std::same_as<Sentinel, ConstSentinel>);

    Sentinel s1(Sent{5}, nullptr);
    ConstSentinel s2 = s1;
    assert(s2.base().i == 5);
  }

  // pred is init correctly
  {
    bool called = false;
    auto pred   = [&](int) {
      called = true;
      return false;
    };

    using R             = cuda::std::ranges::take_while_view<Range, decltype(pred)>;
    using Sentinel      = cuda::std::ranges::sentinel_t<R>;
    using ConstSentinel = cuda::std::ranges::sentinel_t<const R>;
    static_assert(!cuda::std::same_as<Sentinel, ConstSentinel>);

    int i     = 10;
    int* iter = &i;
    Sentinel s1(Sent{0}, &pred);
    ConstSentinel s2 = s1;

    bool b = iter == s2;
    unused(b);
    assert(called);
  }

  // LWG 3708 `take_while_view::sentinel`'s conversion constructor should move
  {
    using R             = cuda::std::ranges::take_while_view<Rng, Pred>;
    using Sentinel      = cuda::std::ranges::sentinel_t<R>;
    using ConstSentinel = cuda::std::ranges::sentinel_t<const R>;
    static_assert(!cuda::std::same_as<Sentinel, ConstSentinel>);

    Sentinel s1(Sent{5}, nullptr);
    ConstSentinel s2 = s1;
    assert(s2.base().i == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
