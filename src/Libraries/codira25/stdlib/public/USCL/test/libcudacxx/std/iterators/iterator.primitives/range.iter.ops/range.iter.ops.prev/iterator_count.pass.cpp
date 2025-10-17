/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

// UNSUPPORTED: msvc-19.16

// ranges::prev(it, n)

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/iterator>
#include <uscl/std/utility>

#include "test_iterators.h"

template <typename It>
__host__ __device__ constexpr void check(int* first, cuda::std::iter_difference_t<It> n, int* expected)
{
  It it(first);
  decltype(auto) result = cuda::std::ranges::prev(cuda::std::move(it), n);
  static_assert(cuda::std::same_as<decltype(result), It>);
  assert(base(result) == expected);
}

__host__ __device__ constexpr bool test()
{
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // Check prev() forward
  for (int n = 0; n != 10; ++n)
  {
    check<bidirectional_iterator<int*>>(range + n, n, range);
    check<random_access_iterator<int*>>(range + n, n, range);
    check<contiguous_iterator<int*>>(range + n, n, range);
    check<int*>(range + n, n, range);
  }

  // Check prev() backward
  for (int n = 0; n != 10; ++n)
  {
    check<bidirectional_iterator<int*>>(range, -n, range + n);
    check<random_access_iterator<int*>>(range, -n, range + n);
    check<contiguous_iterator<int*>>(range, -n, range + n);
    check<int*>(range, -n, range + n);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
