/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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

// UNSUPPORTED: c++17

// <cuda/std/iterator>
//
// reverse_iterator
//
// pointer operator->() const;

#include <uscl/std/iterator>
#include <uscl/std/type_traits>

#include "test_iterators.h"

template <class T>
concept HasArrow = requires(T t) { t.operator->(); };

struct simple_bidirectional_iterator
{
  using iterator_category = cuda::std::bidirectional_iterator_tag;
  using value_type        = int;
  using difference_type   = int;
  using pointer           = int*;
  using reference         = int&;

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;

  __host__ __device__ simple_bidirectional_iterator& operator++();
  __host__ __device__ simple_bidirectional_iterator& operator--();
  __host__ __device__ simple_bidirectional_iterator operator++(int);
  __host__ __device__ simple_bidirectional_iterator operator--(int);

  __host__ __device__ friend bool operator==(const simple_bidirectional_iterator&, const simple_bidirectional_iterator&);
};
static_assert(cuda::std::bidirectional_iterator<simple_bidirectional_iterator>);
static_assert(!cuda::std::random_access_iterator<simple_bidirectional_iterator>);

using PtrRI = cuda::std::reverse_iterator<int*>;
static_assert(HasArrow<PtrRI>);

using PtrLikeRI = cuda::std::reverse_iterator<simple_bidirectional_iterator>;
static_assert(HasArrow<PtrLikeRI>);

// `bidirectional_iterator` from `test_iterators.h` doesn't define `operator->`.
using NonPtrRI = cuda::std::reverse_iterator<bidirectional_iterator<int*>>;
static_assert(!HasArrow<NonPtrRI>);

int main(int, char**)
{
  return 0;
}
