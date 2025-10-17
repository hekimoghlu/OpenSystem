/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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

// template<class T, class Compare>
//   T
//   min(initializer_list<T> t, Compare comp);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>
#include <uscl/std/functional>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int i = cuda::std::min({2, 3, 1}, cuda::std::greater<int>());
  assert(i == 3);
  i = cuda::std::min({2, 1, 3}, cuda::std::greater<int>());
  assert(i == 3);
  i = cuda::std::min({3, 1, 2}, cuda::std::greater<int>());
  assert(i == 3);
  i = cuda::std::min({3, 2, 1}, cuda::std::greater<int>());
  assert(i == 3);
  i = cuda::std::min({1, 2, 3}, cuda::std::greater<int>());
  assert(i == 3);
  i = cuda::std::min({1, 3, 2}, cuda::std::greater<int>());
  assert(i == 3);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
