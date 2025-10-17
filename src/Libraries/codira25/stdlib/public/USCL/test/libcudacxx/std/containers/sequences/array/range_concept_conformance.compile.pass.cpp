/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// array

#include <uscl/std/array>
#include <uscl/std/concepts>
#include <uscl/std/ranges>

using range = cuda::std::array<int, 10>;

static_assert(!cuda::std::ranges::view<range>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<range>, range::iterator>);
static_assert(cuda::std::ranges::common_range<range>);
static_assert(cuda::std::ranges::random_access_range<range>);
static_assert(cuda::std::ranges::contiguous_range<range>);
static_assert(cuda::std::ranges::sized_range<range>);
static_assert(!cuda::std::ranges::borrowed_range<range>);
static_assert(cuda::std::ranges::viewable_range<range>);

static_assert(!cuda::std::ranges::view<range const>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<range const>, range::const_iterator>);
static_assert(cuda::std::ranges::common_range<range const>);
static_assert(cuda::std::ranges::random_access_range<range const>);
static_assert(cuda::std::ranges::contiguous_range<range const>);
static_assert(cuda::std::ranges::sized_range<range const>);
static_assert(!cuda::std::ranges::borrowed_range<range const>);
static_assert(!cuda::std::ranges::viewable_range<range const>);

int main(int, char**)
{
  return 0;
}
