/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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

// template<class R>
// concept input_range;

#include <uscl/std/ranges>

#include "test_iterators.h"
#include "test_range.h"

static_assert(cuda::std::ranges::input_range<test_range<cpp17_input_iterator>>, "");
static_assert(cuda::std::ranges::input_range<test_range<cpp17_input_iterator> const>, "");

static_assert(cuda::std::ranges::input_range<test_range<cpp20_input_iterator>>, "");
static_assert(cuda::std::ranges::input_range<test_range<cpp20_input_iterator> const>, "");

static_assert(cuda::std::ranges::input_range<test_non_const_range<cpp17_input_iterator>>, "");
static_assert(cuda::std::ranges::input_range<test_non_const_range<cpp20_input_iterator>>, "");

static_assert(!cuda::std::ranges::input_range<test_non_const_range<cpp17_input_iterator> const>, "");
static_assert(!cuda::std::ranges::input_range<test_non_const_range<cpp20_input_iterator> const>, "");

static_assert(cuda::std::ranges::input_range<test_common_range<forward_iterator>>, "");
static_assert(!cuda::std::ranges::input_range<test_common_range<cpp20_input_iterator>>, "");

static_assert(cuda::std::ranges::input_range<test_common_range<forward_iterator> const>, "");
static_assert(!cuda::std::ranges::input_range<test_common_range<cpp20_input_iterator> const>, "");

static_assert(cuda::std::ranges::input_range<test_non_const_common_range<forward_iterator>>, "");
static_assert(!cuda::std::ranges::input_range<test_non_const_common_range<cpp20_input_iterator>>, "");

static_assert(!cuda::std::ranges::input_range<test_non_const_common_range<forward_iterator> const>, "");
static_assert(!cuda::std::ranges::input_range<test_non_const_common_range<cpp20_input_iterator> const>, "");

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>*&>, "");
static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>*&&>, "");
static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>* const>, "");
static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>* const&>, "");
static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>* const&&>, "");

static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* [10]>, "");
static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* (&) [10]>, "");
static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* (&&) [10]>, "");
static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* const[10]>, "");
static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* const (&)[10]>, "");
static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* const (&&)[10]>, "");
#endif

int main(int, char**)
{
  return 0;
}
