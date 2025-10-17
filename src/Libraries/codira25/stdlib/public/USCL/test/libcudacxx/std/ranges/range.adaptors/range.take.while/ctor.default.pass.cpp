/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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

// take_while_view() requires default_initializable<V> && default_initializable<Pred> = default;

#include <uscl/std/cassert>
#include <uscl/std/ranges>
#include <uscl/std/type_traits>

template <bool defaultInitable>
struct View : cuda::std::ranges::view_base
{
  int i = 0;
  template <bool defaultInitable2 = defaultInitable, cuda::std::enable_if_t<defaultInitable2, int> = 0>
  __host__ __device__ constexpr explicit View() noexcept {};
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

template <bool defaultInitable>
struct Pred
{
  int i = 0;
  template <bool defaultInitable2 = defaultInitable, cuda::std::enable_if_t<defaultInitable2, int> = 0>
  __host__ __device__ constexpr explicit Pred() noexcept {};
  __host__ __device__ bool operator()(int) const;
};

// clang-format off
static_assert( cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<true >, Pred<true >>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<false>, Pred<true >>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<true >, Pred<false>>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<false>, Pred<false>>>);
// clang-format on

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::take_while_view<View<true>, Pred<true>> twv = {};
    assert(twv.base().i == 0);
    assert(twv.pred().i == 0);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
