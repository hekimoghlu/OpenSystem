/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   constexpr OutIter   // constexpr after C++17
//   copy(InIter first, InIter last, OutIter result);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "copy_common.h"

TEST_CONSTEXPR_CXX20 __host__ __device__ bool test()
{
  test<random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, int*>();

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
