/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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

// <cuda/std/iterator>

// move_iterator

// template<sized_sentinel_for<Iterator> S>
//   friend constexpr iter_difference_t<Iterator>
//     operator-(const move_sentinel<S>& x, const move_iterator& y); // Since C++20
// template<sized_sentinel_for<Iterator> S>
//   friend constexpr iter_difference_t<Iterator>
//     operator-(const move_iterator& x, const move_sentinel<S>& y); // Since C++20

#include <uscl/std/cassert>
#include <uscl/std/iterator>
#include <uscl/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

// The `operator-` calls the underlying iterator and sentinel's `operator-`.

struct CustomIt
{
  using value_type        = int;
  using difference_type   = int;
  using reference         = int&;
  using pointer           = int*;
  using iterator_category = cuda::std::input_iterator_tag;

  CustomIt() = default;
  __host__ __device__ constexpr explicit CustomIt(int* p)
      : p_(p)
  {}
  __host__ __device__ int& operator*() const;
  __host__ __device__ CustomIt& operator++();
  __host__ __device__ CustomIt operator++(int);
  __host__ __device__ constexpr friend difference_type operator-(const CustomIt& a, const CustomIt& b)
  {
    return static_cast<difference_type>(a.p_ - b.p_);
  }
  int* p_ = nullptr;
};

template <class It, class Sent = sized_sentinel<It>>
__host__ __device__ constexpr void test_one()
{
  int arr[] = {3, 1, 4};

  const cuda::std::move_iterator<It> it_a{It(arr)};
  const cuda::std::move_iterator<It> it_b{It(arr + 1)};

  const cuda::std::move_sentinel<Sent> sent_a{Sent(It(arr))};
  const cuda::std::move_sentinel<Sent> sent_b{Sent(It(arr + 1))};
  const cuda::std::move_sentinel<Sent> sent_c{Sent(It(arr + 2))};

  static_assert(cuda::std::is_same_v<decltype(it_a - sent_a), cuda::std::iter_difference_t<It>>);
  static_assert(cuda::std::is_same_v<decltype(sent_a - it_a), cuda::std::iter_difference_t<It>>);

  // it_a
  assert(it_a - sent_a == 0);
  assert(sent_a - it_a == 0);

  assert(it_a - sent_b == -1);
  assert(sent_b - it_a == 1);

  assert(it_a - sent_c == -2);
  assert(sent_c - it_a == 2);

  // it_b
  assert(it_b - sent_a == 1);
  assert(sent_a - it_b == -1);

  assert(it_b - sent_b == 0);
  assert(sent_b - it_b == 0);

  assert(it_b - sent_c == -1);
  assert(sent_c - it_b == 1);
}

__host__ __device__ constexpr bool test()
{
  test_one<CustomIt>();
  test_one<cpp17_input_iterator<int*>>();
  test_one<forward_iterator<int*>>();
  test_one<bidirectional_iterator<int*>>();
  test_one<random_access_iterator<int*>>();
  test_one<int*>();
  test_one<const int*>();
  test_one<contiguous_iterator<int*>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
