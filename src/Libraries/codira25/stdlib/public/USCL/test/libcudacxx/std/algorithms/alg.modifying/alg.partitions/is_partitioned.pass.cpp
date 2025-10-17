/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class InputIterator, class Predicate>
//     constexpr bool       // constexpr after C++17
//     is_partitioned(InputIterator first, InputIterator last, Predicate pred);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/functional>

#include "counting_predicates.h"
#include "test_iterators.h"
#include "test_macros.h"

struct is_odd
{
  __host__ __device__ constexpr bool operator()(const int& i) const
  {
    return i & 1;
  }
};

__host__ __device__ constexpr bool test()
{
  {
    const int ia[] = {1, 2, 3, 4, 5, 6};
    unary_counting_predicate<is_odd, int> pred((is_odd()));
    assert(!cuda::std::is_partitioned(cpp17_input_iterator<const int*>(cuda::std::begin(ia)),
                                      cpp17_input_iterator<const int*>(cuda::std::end(ia)),
                                      pred));
    assert(
      static_cast<cuda::std::ptrdiff_t>(pred.count()) <= cuda::std::distance(cuda::std::begin(ia), cuda::std::end(ia)));
  }
  {
    const int ia[] = {1, 3, 5, 2, 4, 6};
    unary_counting_predicate<is_odd, int> pred((is_odd()));
    assert(cuda::std::is_partitioned(cpp17_input_iterator<const int*>(cuda::std::begin(ia)),
                                     cpp17_input_iterator<const int*>(cuda::std::end(ia)),
                                     pred));
    assert(
      static_cast<cuda::std::ptrdiff_t>(pred.count()) <= cuda::std::distance(cuda::std::begin(ia), cuda::std::end(ia)));
  }
  {
    const int ia[] = {2, 4, 6, 1, 3, 5};
    unary_counting_predicate<is_odd, int> pred((is_odd()));
    assert(!cuda::std::is_partitioned(cpp17_input_iterator<const int*>(cuda::std::begin(ia)),
                                      cpp17_input_iterator<const int*>(cuda::std::end(ia)),
                                      pred));
    assert(
      static_cast<cuda::std::ptrdiff_t>(pred.count()) <= cuda::std::distance(cuda::std::begin(ia), cuda::std::end(ia)));
  }
  {
    const int ia[] = {1, 3, 5, 2, 4, 6, 7};
    unary_counting_predicate<is_odd, int> pred((is_odd()));
    assert(!cuda::std::is_partitioned(cpp17_input_iterator<const int*>(cuda::std::begin(ia)),
                                      cpp17_input_iterator<const int*>(cuda::std::end(ia)),
                                      pred));
    assert(
      static_cast<cuda::std::ptrdiff_t>(pred.count()) <= cuda::std::distance(cuda::std::begin(ia), cuda::std::end(ia)));
  }
  {
    const int ia[] = {1, 3, 5, 2, 4, 6, 7};
    unary_counting_predicate<is_odd, int> pred((is_odd()));
    assert(cuda::std::is_partitioned(cpp17_input_iterator<const int*>(cuda::std::begin(ia)),
                                     cpp17_input_iterator<const int*>(cuda::std::begin(ia)),
                                     pred));
    assert(static_cast<cuda::std::ptrdiff_t>(pred.count())
           <= cuda::std::distance(cuda::std::begin(ia), cuda::std::begin(ia)));
  }
  {
    const int ia[] = {1, 3, 5, 7, 9, 11, 2};
    unary_counting_predicate<is_odd, int> pred((is_odd()));
    assert(cuda::std::is_partitioned(cpp17_input_iterator<const int*>(cuda::std::begin(ia)),
                                     cpp17_input_iterator<const int*>(cuda::std::end(ia)),
                                     pred));
    assert(
      static_cast<cuda::std::ptrdiff_t>(pred.count()) <= cuda::std::distance(cuda::std::begin(ia), cuda::std::end(ia)));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
