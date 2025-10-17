/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++17

// template<class I1, class I2, class Out,
//     class R = ranges::less, class P1 = identity, class P2 = identity>
//   concept mergeable = see below;                           // since C++20

#include <uscl/std/iterator>

#include "test_macros.h"

template <class I1, class I2, class O>
__host__ __device__ void test_subsumption()
  requires cuda::std::input_iterator<I1> && cuda::std::input_iterator<I2>;

template <class I1, class I2, class O>
__host__ __device__ void test_subsumption()
  requires cuda::std::weakly_incrementable<O>;

template <class I1, class I2, class O>
__host__ __device__ void test_subsumption()
  requires cuda::std::indirectly_copyable<I1, O> && cuda::std::indirectly_copyable<I2, O>;

template <class I1, class I2, class O>
__host__ __device__ void test_subsumption()
  requires cuda::std::indirect_strict_weak_order<I1, I2>;

template <class I1, class I2, class O>
__host__ __device__ constexpr bool test_subsumption()
  requires cuda::std::mergeable<I1, I2, O>
{
  return true;
}

static_assert(test_subsumption<int*, int*, int*>(), "");

int main(int, char**)
{
  return 0;
}
