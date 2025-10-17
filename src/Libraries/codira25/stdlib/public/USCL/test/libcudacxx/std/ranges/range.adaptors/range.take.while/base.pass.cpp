/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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

// constexpr V base() const & requires copy_constructible<V> { return base_; }
// constexpr V base() && { return cuda::std::move(base_); }

#include <uscl/std/cassert>
#include <uscl/std/ranges>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"
#include "types.h"

struct View : cuda::std::ranges::view_interface<View>
{
  int i;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct MoveOnlyView : View
{
  MoveOnly mo;
};

template <class T>
_CCCL_CONCEPT HasBase = _CCCL_REQUIRES_EXPR((T), T&& t)((cuda::std::forward<T>(t).base()));

struct Pred
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i > 5;
  }
};

static_assert(HasBase<cuda::std::ranges::take_while_view<View, Pred> const&>);
static_assert(HasBase<cuda::std::ranges::take_while_view<View, Pred>&&>);

static_assert(!HasBase<cuda::std::ranges::take_while_view<MoveOnlyView, Pred> const&>);
static_assert(HasBase<cuda::std::ranges::take_while_view<MoveOnlyView, Pred>&&>);

__host__ __device__ constexpr bool test()
{
  // const &
  {
    const cuda::std::ranges::take_while_view<View, Pred> twv{View{{}, 5}, {}};
    decltype(auto) v = twv.base();
    static_assert(cuda::std::same_as<decltype(v), View>);
    assert(v.i == 5);
  }

  // &
  {
    cuda::std::ranges::take_while_view<View, Pred> twv{View{{}, 5}, {}};
    decltype(auto) v = twv.base();
    static_assert(cuda::std::same_as<decltype(v), View>);
    assert(v.i == 5);
  }

  // &&
  {
    cuda::std::ranges::take_while_view<View, Pred> twv{View{{}, 5}, {}};
    decltype(auto) v = cuda::std::move(twv).base();
    static_assert(cuda::std::same_as<decltype(v), View>);
    assert(v.i == 5);
  }

  // const &&
  {
    const cuda::std::ranges::take_while_view<View, Pred> twv{View{{}, 5}, {}};
    decltype(auto) v = cuda::std::move(twv).base();
    static_assert(cuda::std::same_as<decltype(v), View>);
    assert(v.i == 5);
  }

  // move only
  {
    cuda::std::ranges::take_while_view<MoveOnlyView, Pred> twv{MoveOnlyView{{}, 5}, {}};
    decltype(auto) v = cuda::std::move(twv).base();
    static_assert(cuda::std::same_as<decltype(v), MoveOnlyView>);
    assert(v.mo.get() == 5);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
