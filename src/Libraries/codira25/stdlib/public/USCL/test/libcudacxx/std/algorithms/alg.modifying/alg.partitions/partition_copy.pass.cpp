/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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

// template <class InputIterator, class OutputIterator1,
//           class OutputIterator2, class Predicate>
//     constexpr pair<OutputIterator1, OutputIterator2>     // constexpr after C++17
//     partition_copy(InputIterator first, InputIterator last,
//                    OutputIterator1 out_true, OutputIterator2 out_false,
//                    Predicate pred);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

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
    const int ia[] = {1, 2, 3, 4, 6, 8, 5, 7};
    int r1[10]     = {0};
    int r2[10]     = {0};
    typedef cuda::std::pair<cpp17_output_iterator<int*>, int*> P;
    P p = cuda::std::partition_copy(
      cpp17_input_iterator<const int*>(cuda::std::begin(ia)),
      cpp17_input_iterator<const int*>(cuda::std::end(ia)),
      cpp17_output_iterator<int*>(r1),
      r2,
      is_odd());
    assert(base(p.first) == r1 + 4);
    assert(r1[0] == 1);
    assert(r1[1] == 3);
    assert(r1[2] == 5);
    assert(r1[3] == 7);
    assert(p.second == r2 + 4);
    assert(r2[0] == 2);
    assert(r2[1] == 4);
    assert(r2[2] == 6);
    assert(r2[3] == 8);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
