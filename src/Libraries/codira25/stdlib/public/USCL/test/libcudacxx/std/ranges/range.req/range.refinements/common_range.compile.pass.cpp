/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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
// concept common_range;

#include <uscl/std/ranges>

#include "test_iterators.h"

template <class It>
struct Common
{
  __host__ __device__ It begin() const;
  __host__ __device__ It end() const;
};
template <class It>
struct NonCommon
{
  __host__ __device__ It begin() const;
  __host__ __device__ sentinel_wrapper<It> end() const;
};
template <class It, class Sent>
struct Range
{
  __host__ __device__ It begin() const;
  __host__ __device__ Sent end() const;
};

static_assert(!cuda::std::ranges::common_range<Common<cpp17_input_iterator<int*>>>, ""); // not a sentinel for itself
static_assert(!cuda::std::ranges::common_range<Common<cpp20_input_iterator<int*>>>, ""); // not a sentinel for itself
static_assert(cuda::std::ranges::common_range<Common<forward_iterator<int*>>>, "");
static_assert(cuda::std::ranges::common_range<Common<bidirectional_iterator<int*>>>, "");
static_assert(cuda::std::ranges::common_range<Common<random_access_iterator<int*>>>, "");
static_assert(cuda::std::ranges::common_range<Common<contiguous_iterator<int*>>>, "");
static_assert(cuda::std::ranges::common_range<Common<int*>>, "");

static_assert(!cuda::std::ranges::common_range<NonCommon<cpp17_input_iterator<int*>>>, "");
static_assert(!cuda::std::ranges::common_range<NonCommon<cpp20_input_iterator<int*>>>, "");
static_assert(!cuda::std::ranges::common_range<NonCommon<forward_iterator<int*>>>, "");
static_assert(!cuda::std::ranges::common_range<NonCommon<bidirectional_iterator<int*>>>, "");
static_assert(!cuda::std::ranges::common_range<NonCommon<random_access_iterator<int*>>>, "");
static_assert(!cuda::std::ranges::common_range<NonCommon<contiguous_iterator<int*>>>, "");
static_assert(!cuda::std::ranges::common_range<NonCommon<int*>>, "");

// Test when begin() and end() only differ by their constness.
static_assert(!cuda::std::ranges::common_range<Range<int*, int const*>>, "");

// Simple test with a sized_sentinel.
static_assert(!cuda::std::ranges::common_range<Range<int*, sized_sentinel<int*>>>, "");

// Make sure cv-qualification doesn't impact the concept when begin() and end() have matching qualifiers.
static_assert(cuda::std::ranges::common_range<Common<forward_iterator<int*>> const>, "");
static_assert(!cuda::std::ranges::common_range<NonCommon<forward_iterator<int*>> const>, "");

// Test with a range that's a common_range only when const-qualified.
struct Range1
{
  __host__ __device__ int* begin();
  __host__ __device__ int const* begin() const;
  __host__ __device__ int const* end() const;
};
static_assert(!cuda::std::ranges::common_range<Range1>, "");
static_assert(cuda::std::ranges::common_range<Range1 const>, "");

// Test with a range that's a common_range only when not const-qualified.
struct Range2
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end();
  __host__ __device__ int const* end() const;
};
static_assert(cuda::std::ranges::common_range<Range2>, "");
static_assert(!cuda::std::ranges::common_range<Range2 const>, "");

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

static_assert(!cuda::std::ranges::common_range<Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::common_range<Holder<Incomplete>*&>, "");
static_assert(!cuda::std::ranges::common_range<Holder<Incomplete>*&&>, "");
static_assert(!cuda::std::ranges::common_range<Holder<Incomplete>* const>, "");
static_assert(!cuda::std::ranges::common_range<Holder<Incomplete>* const&>, "");
static_assert(!cuda::std::ranges::common_range<Holder<Incomplete>* const&&>, "");

static_assert(cuda::std::ranges::common_range<Holder<Incomplete>* [10]>, "");
static_assert(cuda::std::ranges::common_range<Holder<Incomplete>* (&) [10]>, "");
static_assert(cuda::std::ranges::common_range<Holder<Incomplete>* (&&) [10]>, "");
static_assert(cuda::std::ranges::common_range<Holder<Incomplete>* const[10]>, "");
static_assert(cuda::std::ranges::common_range<Holder<Incomplete>* const (&)[10]>, "");
static_assert(cuda::std::ranges::common_range<Holder<Incomplete>* const (&&)[10]>, "");
#endif

int main(int, char**)
{
  return 0;
}
