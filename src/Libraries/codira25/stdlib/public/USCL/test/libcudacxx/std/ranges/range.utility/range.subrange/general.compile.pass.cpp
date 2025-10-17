/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER > 2017
template <cuda::std::ranges::subrange_kind K, class... Args>
concept ValidSubrangeKind = requires { typename cuda::std::ranges::subrange<Args..., K>; };

template <class... Args>
concept ValidSubrange = requires { typename cuda::std::ranges::subrange<Args...>; };
#else
// clang is really not helpful here failing with concept emulation
template <class It, class = void>
constexpr bool ValidSubrange1 = false;

template <class It>
constexpr bool ValidSubrange1<It, cuda::std::void_t<cuda::std::ranges::subrange<It>>> = true;

template <class It, class Sent, class = void>
constexpr bool ValidSubrange2 = false;

template <class It, class Sent>
constexpr bool ValidSubrange2<It, Sent, cuda::std::void_t<cuda::std::ranges::subrange<It, Sent>>> = true;

template <class...>
constexpr bool ValidSubrange = false;

template <class It>
constexpr bool ValidSubrange<It> = ValidSubrange1<It>;

template <class It, class Sent>
constexpr bool ValidSubrange<It, Sent> = ValidSubrange2<It, Sent>;

template <cuda::std::ranges::subrange_kind Kind, class It, class Sent, class = void>
constexpr bool ValidSubrangeKind = false;

template <cuda::std::ranges::subrange_kind Kind, class It, class Sent>
constexpr bool ValidSubrangeKind<Kind, It, Sent, cuda::std::void_t<cuda::std::ranges::subrange<It, Sent, Kind>>> = true;
#endif

static_assert(ValidSubrange<forward_iterator<int*>>);
static_assert(ValidSubrange<forward_iterator<int*>, forward_iterator<int*>>);
static_assert(
  ValidSubrangeKind<cuda::std::ranges::subrange_kind::unsized, forward_iterator<int*>, forward_iterator<int*>>);
static_assert(
  ValidSubrangeKind<cuda::std::ranges::subrange_kind::sized, forward_iterator<int*>, forward_iterator<int*>>);
// Wrong sentinel type.
static_assert(!ValidSubrange<forward_iterator<int*>, int*>);
static_assert(ValidSubrange<int*>);
static_assert(ValidSubrange<int*, int*>);
// Must be sized.
static_assert(!ValidSubrangeKind<cuda::std::ranges::subrange_kind::unsized, int*, int*>);
static_assert(ValidSubrangeKind<cuda::std::ranges::subrange_kind::sized, int*, int*>);
// Wrong sentinel type.
static_assert(!ValidSubrange<int*, forward_iterator<int*>>);
// Not an iterator.
static_assert(!ValidSubrange<int>);

int main(int, char**)
{
  return 0;
}
