/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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

// template<class T>
// using iterator_t = decltype(ranges::begin(declval<T&>()));

#include <uscl/std/concepts>
#include <uscl/std/ranges>

#include "test_range.h"

static_assert(
  cuda::std::same_as<cuda::std::ranges::iterator_t<test_range<cpp17_input_iterator>>, cpp17_input_iterator<int*>>, "");
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_range<cpp17_input_iterator> const>,
                                 cpp17_input_iterator<int const*>>,
              "");

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_non_const_range<cpp17_input_iterator>>,
                                 cpp17_input_iterator<int*>>,
              "");

static_assert(
  cuda::std::same_as<cuda::std::ranges::iterator_t<test_common_range<cpp17_input_iterator>>, cpp17_input_iterator<int*>>,
  "");
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_common_range<cpp17_input_iterator> const>,
                                 cpp17_input_iterator<int const*>>,
              "");

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_non_const_common_range<cpp17_input_iterator>>,
                                 cpp17_input_iterator<int*>>,
              "");

int main(int, char**)
{
  return 0;
}
