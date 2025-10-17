/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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

// template<class I1, class I2, class R, class P1, class P2>
// concept indirectly_comparable;

#include <uscl/std/functional>
#include <uscl/std/iterator>
#include <uscl/std/type_traits>

template <class F>
  requires cuda::std::indirectly_comparable<int*, char*, F>
        && true // This true is an additional atomic constraint as a tie breaker
__host__ __device__ constexpr bool subsumes(F)
{
  return true;
}

template <class F>
  requires cuda::std::indirect_binary_predicate<F,
                                                cuda::std::projected<int*, cuda::std::identity>,
                                                cuda::std::projected<char*, cuda::std::identity>>
__host__ __device__ void subsumes(F);

template <class F>
  requires cuda::std::indirect_binary_predicate<F,
                                                cuda::std::projected<int*, cuda::std::identity>,
                                                cuda::std::projected<char*, cuda::std::identity>>
        && true // This true is an additional atomic constraint as a tie breaker
__host__ __device__ constexpr bool is_subsumed(F)
{
  return true;
}

template <class F>
  requires cuda::std::indirectly_comparable<int*, char*, F>
__host__ __device__ void is_subsumed(F);

static_assert(subsumes(cuda::std::less<int>()), "");
static_assert(is_subsumed(cuda::std::less<int>()), "");

int main(int, char**)
{
  return 0;
}
