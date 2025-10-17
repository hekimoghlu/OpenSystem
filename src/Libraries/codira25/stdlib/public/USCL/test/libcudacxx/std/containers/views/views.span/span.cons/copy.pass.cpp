/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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

// <span>

//  constexpr span(const span& other) noexcept = default;

#include <uscl/std/cassert>
#include <uscl/std/span>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr bool doCopy(const T& rhs)
{
  static_assert(noexcept(T{rhs}));
  T lhs{rhs};
  return lhs.data() == rhs.data() && lhs.size() == rhs.size();
}

struct A
{};

template <typename T>
__host__ __device__ void testCV()
{
  int arr[] = {1, 2, 3};
  assert((doCopy(cuda::std::span<T>())));
  assert((doCopy(cuda::std::span<T, 0>())));
  assert((doCopy(cuda::std::span<T>(&arr[0], 1))));
  assert((doCopy(cuda::std::span<T, 1>(&arr[0], 1))));
  assert((doCopy(cuda::std::span<T>(&arr[0], 2))));
  assert((doCopy(cuda::std::span<T, 2>(&arr[0], 2))));
}

TEST_GLOBAL_VARIABLE constexpr int carr[] = {1, 2, 3};

int main(int, char**)
{
  static_assert(doCopy(cuda::std::span<int>()));
  static_assert(doCopy(cuda::std::span<int, 0>()));
  static_assert(doCopy(cuda::std::span<const int>(&carr[0], 1)));
  static_assert(doCopy(cuda::std::span<const int, 1>(&carr[0], 1)));
  static_assert(doCopy(cuda::std::span<const int>(&carr[0], 2)));
  static_assert(doCopy(cuda::std::span<const int, 2>(&carr[0], 2)));

  static_assert(doCopy(cuda::std::span<long>()));
  static_assert(doCopy(cuda::std::span<double>()));
  static_assert(doCopy(cuda::std::span<A>()));

  testCV<int>();
  testCV<const int>();
  testCV<volatile int>();
  testCV<const volatile int>();

  return 0;
}
