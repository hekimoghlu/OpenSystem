/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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

// template<class In>
// concept input_or_output_iterator;

#include <uscl/std/iterator>

#include "test_iterators.h"

static_assert(cuda::std::input_or_output_iterator<int*>, "");
static_assert(cuda::std::input_or_output_iterator<int const*>, "");
static_assert(cuda::std::input_or_output_iterator<int volatile*>, "");
static_assert(cuda::std::input_or_output_iterator<int const volatile*>, "");

static_assert(cuda::std::input_or_output_iterator<cpp17_input_iterator<int*>>, "");
static_assert(cuda::std::input_or_output_iterator<cpp17_input_iterator<int const*>>, "");
static_assert(cuda::std::input_or_output_iterator<cpp17_input_iterator<int volatile*>>, "");
static_assert(cuda::std::input_or_output_iterator<cpp17_input_iterator<int const volatile*>>, "");

static_assert(cuda::std::input_or_output_iterator<forward_iterator<int*>>, "");
static_assert(cuda::std::input_or_output_iterator<forward_iterator<int const*>>, "");
static_assert(cuda::std::input_or_output_iterator<forward_iterator<int volatile*>>, "");
static_assert(cuda::std::input_or_output_iterator<forward_iterator<int const volatile*>>, "");

static_assert(cuda::std::input_or_output_iterator<bidirectional_iterator<int*>>, "");
static_assert(cuda::std::input_or_output_iterator<bidirectional_iterator<int const*>>, "");
static_assert(cuda::std::input_or_output_iterator<bidirectional_iterator<int volatile*>>, "");
static_assert(cuda::std::input_or_output_iterator<bidirectional_iterator<int const volatile*>>, "");

static_assert(cuda::std::input_or_output_iterator<random_access_iterator<int*>>, "");
static_assert(cuda::std::input_or_output_iterator<random_access_iterator<int const*>>, "");
static_assert(cuda::std::input_or_output_iterator<random_access_iterator<int volatile*>>, "");
static_assert(cuda::std::input_or_output_iterator<random_access_iterator<int const volatile*>>, "");

static_assert(!cuda::std::input_or_output_iterator<void*>, "");
static_assert(!cuda::std::input_or_output_iterator<void const*>, "");
static_assert(!cuda::std::input_or_output_iterator<void volatile*>, "");
static_assert(!cuda::std::input_or_output_iterator<void const volatile*>, "");

struct S
{};
static_assert(!cuda::std::input_or_output_iterator<S>, "");
static_assert(!cuda::std::input_or_output_iterator<int S::*>, "");
static_assert(!cuda::std::input_or_output_iterator<int (S::*)()>, "");
static_assert(!cuda::std::input_or_output_iterator<int (S::*)() const>, "");
static_assert(!cuda::std::input_or_output_iterator<int (S::*)() volatile>, "");

struct missing_dereference
{
  using difference_type = cuda::std::ptrdiff_t;

  __host__ __device__ missing_dereference& operator++();
  __host__ __device__ missing_dereference& operator++(int);
};
static_assert(cuda::std::weakly_incrementable<missing_dereference>
                && !cuda::std::input_or_output_iterator<missing_dereference>,
              "");

struct void_dereference
{
  using difference_type = cuda::std::ptrdiff_t;

  __host__ __device__ void operator*();
  __host__ __device__ void_dereference& operator++();
  __host__ __device__ void_dereference& operator++(int);
};
static_assert(
  cuda::std::weakly_incrementable<void_dereference> && !cuda::std::input_or_output_iterator<void_dereference>, "");

struct not_weakly_incrementable
{
  __host__ __device__ int operator*() const;
};
static_assert(!cuda::std::input_or_output_iterator<not_weakly_incrementable>, "");

int main(int, char**)
{
  return 0;
}
