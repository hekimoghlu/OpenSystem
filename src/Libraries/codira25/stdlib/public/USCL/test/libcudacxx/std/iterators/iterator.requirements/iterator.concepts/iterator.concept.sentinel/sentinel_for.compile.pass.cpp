/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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

// template<class S, class I>
// concept sentinel_for;

#include <uscl/std/iterator>

#include "test_macros.h"

static_assert(cuda::std::sentinel_for<int*, int*>, "");
static_assert(!cuda::std::sentinel_for<int*, long*>, "");
struct nth_element_sentinel
{
  __host__ __device__ friend bool operator==(const nth_element_sentinel&, int*);
  __host__ __device__ friend bool operator==(int*, const nth_element_sentinel&);
  __host__ __device__ friend bool operator!=(const nth_element_sentinel&, int*);
  __host__ __device__ friend bool operator!=(int*, const nth_element_sentinel&);
};
static_assert(cuda::std::sentinel_for<nth_element_sentinel, int*>, "");

struct not_semiregular
{
  not_semiregular() = delete;
  __host__ __device__ friend bool operator==(const not_semiregular&, int*);
  __host__ __device__ friend bool operator==(int*, const not_semiregular&);
  __host__ __device__ friend bool operator!=(const not_semiregular&, int*);
  __host__ __device__ friend bool operator!=(int*, const not_semiregular&);
};
static_assert(!cuda::std::sentinel_for<not_semiregular, int*>, "");

struct weakly_equality_comparable_with_int
{
  __host__ __device__ friend bool operator==(const weakly_equality_comparable_with_int&, int);
  __host__ __device__ friend bool operator==(int, const weakly_equality_comparable_with_int&);
  __host__ __device__ friend bool operator!=(const weakly_equality_comparable_with_int&, int*);
  __host__ __device__ friend bool operator!=(int*, const weakly_equality_comparable_with_int&);
};
static_assert(!cuda::std::sentinel_for<weakly_equality_comparable_with_int, int>, "");

struct move_only_iterator
{
  using value_type      = int;
  using difference_type = cuda::std::ptrdiff_t;

  move_only_iterator() = default;

  move_only_iterator(move_only_iterator&&)            = default;
  move_only_iterator& operator=(move_only_iterator&&) = default;

  move_only_iterator(move_only_iterator const&)            = delete;
  move_only_iterator& operator=(move_only_iterator const&) = delete;

  __host__ __device__ value_type operator*() const;
  __host__ __device__ move_only_iterator& operator++();
  __host__ __device__ move_only_iterator operator++(int);

  __host__ __device__ bool operator==(move_only_iterator const&) const;
  __host__ __device__ bool operator!=(move_only_iterator const&) const;
};

static_assert(cuda::std::movable<move_only_iterator> && !cuda::std::copyable<move_only_iterator>
                && cuda::std::input_or_output_iterator<move_only_iterator>
                && !cuda::std::sentinel_for<move_only_iterator, move_only_iterator>,
              "");

int main(int, char**)
{
  return 0;
}
