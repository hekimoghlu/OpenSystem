/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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

// UNSUPPORTED: msvc-19.16

// template<class I1, class I2, class R, class P1, class P2>
// concept indirectly_comparable;

#include <uscl/std/functional>
#include <uscl/std/iterator>
#include <uscl/std/type_traits>

struct Deref
{
  __host__ __device__ int operator()(int*) const;
};

static_assert(!cuda::std::indirectly_comparable<int, int, cuda::std::less<int>>, ""); // not dereferenceable
static_assert(!cuda::std::indirectly_comparable<int*, int*, int>, ""); // not a predicate
static_assert(cuda::std::indirectly_comparable<int*, int*, cuda::std::less<int>>, "");
static_assert(!cuda::std::indirectly_comparable<int**, int*, cuda::std::less<int>>, "");
static_assert(cuda::std::indirectly_comparable<int**, int*, cuda::std::less<int>, Deref>, "");
static_assert(!cuda::std::indirectly_comparable<int**, int*, cuda::std::less<int>, Deref, Deref>, "");
static_assert(!cuda::std::indirectly_comparable<int**, int*, cuda::std::less<int>, cuda::std::identity, Deref>, "");
static_assert(cuda::std::indirectly_comparable<int*, int**, cuda::std::less<int>, cuda::std::identity, Deref>, "");

int main(int, char**)
{
  return 0;
}
