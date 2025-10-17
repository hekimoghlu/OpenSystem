/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

// template<ForwardIterator Iter>
//   requires LessThanComparable<Iter::value_type>
//   Iter
//   min_element(Iter first, Iter last);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "cases.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
__host__ __device__ constexpr void test(const int (&input_data)[num_elements])
{
  Iter first{cuda::std::begin(input_data)};
  Iter last{cuda::std::end(input_data)};

  Iter i = cuda::std::min_element(first, last);
  if (first != last)
  {
    for (Iter j = first; j != last; ++j)
    {
      assert(!(*j < *i));
    }
  }
  else
  {
    assert(i == last);
  }
}

__host__ __device__ constexpr bool test()
{
  constexpr int input_data[num_elements] = INPUT_DATA;
  test<forward_iterator<const int*>>(input_data);
  test<bidirectional_iterator<const int*>>(input_data);
  test<random_access_iterator<const int*>>(input_data);
  test<const int*>(input_data);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
