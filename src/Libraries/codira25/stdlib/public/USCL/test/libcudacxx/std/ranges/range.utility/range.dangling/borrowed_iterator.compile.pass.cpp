/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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

// cuda::std::ranges::borrowed_iterator_t;

#include <uscl/std/concepts>
#include <uscl/std/ranges>
#include <uscl/std/span>
#include <uscl/std/string_view>

#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>
#endif // _LIBCUDACXX_HAS_STRING
#include <uscl/std/inplace_vector>
#include <uscl/std/string_view>

#if defined(_LIBCUDACXX_HAS_STRING)
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_iterator_t<cuda::std::string>, cuda::std::ranges::dangling>);
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_iterator_t<cuda::std::string&&>, cuda::std::ranges::dangling>);
#endif
static_assert(cuda::std::same_as<cuda::std::ranges::borrowed_iterator_t<cuda::std::inplace_vector<int, 3>>,
                                 cuda::std::ranges::dangling>);

#if defined(_LIBCUDACXX_HAS_STRING)
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_iterator_t<cuda::std::string&>, cuda::std::string::iterator>);
#endif
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_iterator_t<cuda::std::string_view>, cuda::std::string_view::iterator>);
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_iterator_t<cuda::std::span<int>>, cuda::std::span<int>::iterator>);

#if TEST_STD_VER >= 2020
template <class T>
constexpr bool has_borrowed_iterator = requires { typename cuda::std::ranges::borrowed_iterator_t<T>; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
constexpr bool has_borrowed_iterator = false;

template <class T>
constexpr bool has_borrowed_iterator<T, cuda::std::void_t<cuda::std::ranges::borrowed_iterator_t<T>>> = true;
#endif // TEST_STD_VER <= 2017

static_assert(!has_borrowed_iterator<int>);

int main(int, char**)
{
  return 0;
}
