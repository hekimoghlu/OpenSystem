/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// ranges::next
// Make sure we're SFINAE-friendly when the template argument constraints are not met.

#include <uscl/std/cstddef>
#include <uscl/std/iterator>
#include <uscl/std/utility>

#include "test_iterators.h"

#if TEST_STD_VER > 2017
template <class... Args>
concept has_ranges_next = requires(Args&&... args) {
  { cuda::std::ranges::next(cuda::std::forward<Args>(args)...) };
};
#else
template <class... Args>
constexpr bool has_ranges_next = cuda::std::invocable<cuda::std::ranges::__next::__fn, Args...>;
#endif

class not_incrementable
{};
static_assert(!has_ranges_next<not_incrementable>);
static_assert(!has_ranges_next<not_incrementable, cuda::std::ptrdiff_t>);
static_assert(!has_ranges_next<not_incrementable, not_incrementable>);
static_assert(!has_ranges_next<not_incrementable, cuda::std::ptrdiff_t, not_incrementable>);

// Test the test
using It2 = forward_iterator<int*>;
static_assert(has_ranges_next<It2>);
static_assert(has_ranges_next<It2, cuda::std::ptrdiff_t>);
static_assert(has_ranges_next<It2, cuda::std::ptrdiff_t, It2>);

int main(int, char**)
{
  return 0;
}
