/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 9, 2022.
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

// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

using FI = forward_iterator<int*>;
TEST_GLOBAL_VARIABLE FI fi{nullptr};
TEST_GLOBAL_VARIABLE int* ptr = nullptr;

static_assert(cuda::std::same_as<decltype(cuda::std::ranges::subrange(fi, fi)),
                                 cuda::std::ranges::subrange<FI, FI, cuda::std::ranges::subrange_kind::unsized>>);
static_assert(cuda::std::same_as<decltype(cuda::std::ranges::subrange(ptr, ptr, 0)),
                                 cuda::std::ranges::subrange<int*, int*, cuda::std::ranges::subrange_kind::sized>>);
static_assert(
  cuda::std::same_as<decltype(cuda::std::ranges::subrange(ptr, nullptr, 0)),
                     cuda::std::ranges::subrange<int*, cuda::std::nullptr_t, cuda::std::ranges::subrange_kind::sized>>);

struct ForwardRange
{
  __host__ __device__ forward_iterator<int*> begin() const;
  __host__ __device__ forward_iterator<int*> end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<ForwardRange> = true;

struct SizedRange
{
  __host__ __device__ int* begin();
  __host__ __device__ int* end();
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<SizedRange> = true;

static_assert(cuda::std::same_as<decltype(cuda::std::ranges::subrange(ForwardRange())),
                                 cuda::std::ranges::subrange<FI, FI, cuda::std::ranges::subrange_kind::unsized>>);
static_assert(cuda::std::same_as<decltype(cuda::std::ranges::subrange(SizedRange())),
                                 cuda::std::ranges::subrange<int*, int*, cuda::std::ranges::subrange_kind::sized>>);
static_assert(cuda::std::same_as<decltype(cuda::std::ranges::subrange(SizedRange(), 8)),
                                 cuda::std::ranges::subrange<int*, int*, cuda::std::ranges::subrange_kind::sized>>);

int main(int, char**)
{
  unused(fi);
  unused(ptr);

  return 0;
}
