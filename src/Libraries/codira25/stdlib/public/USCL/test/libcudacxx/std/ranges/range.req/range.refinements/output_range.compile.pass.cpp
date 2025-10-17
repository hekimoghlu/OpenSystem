/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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

// template<class R, class T>
// concept output_range;

#include <uscl/std/iterator>
#include <uscl/std/ranges>

#include "test_iterators.h"
#include "test_range.h"

struct T
{};

// Satisfied when it's a range and has the right iterator
struct GoodRange
{
  __host__ __device__ cpp17_output_iterator<T*> begin();
  __host__ __device__ sentinel end();
};
static_assert(cuda::std::ranges::range<GoodRange>, "");
static_assert(cuda::std::output_iterator<cuda::std::ranges::iterator_t<GoodRange>, T>, "");
static_assert(cuda::std::ranges::output_range<GoodRange, T>, "");

// Not satisfied when it's not a range
struct NotRange
{
  __host__ __device__ cpp17_output_iterator<T*> begin();
};
static_assert(!cuda::std::ranges::range<NotRange>, "");
static_assert(cuda::std::output_iterator<cuda::std::ranges::iterator_t<NotRange>, T>, "");
static_assert(!cuda::std::ranges::output_range<NotRange, T>, "");

// Not satisfied when the iterator is not an output_iterator
struct RangeWithBadIterator
{
  __host__ __device__ cpp17_input_iterator<T const*> begin();
  __host__ __device__ sentinel end();
};
static_assert(cuda::std::ranges::range<RangeWithBadIterator>, "");
static_assert(!cuda::std::output_iterator<cuda::std::ranges::iterator_t<RangeWithBadIterator>, T>, "");
static_assert(!cuda::std::ranges::output_range<RangeWithBadIterator, T>, "");

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>*, Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>*&, Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>*&&, Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const, Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const&, Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const&&, Holder<Incomplete>*>, "");

static_assert(cuda::std::ranges::output_range<Holder<Incomplete>* [10], Holder<Incomplete>*>, "");
static_assert(cuda::std::ranges::output_range<Holder<Incomplete>* (&) [10], Holder<Incomplete>*>, "");
static_assert(cuda::std::ranges::output_range<Holder<Incomplete>* (&&) [10], Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const[10], Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const (&)[10], Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const (&&)[10], Holder<Incomplete>*>, "");
#endif

int main(int, char**)
{
  return 0;
}
