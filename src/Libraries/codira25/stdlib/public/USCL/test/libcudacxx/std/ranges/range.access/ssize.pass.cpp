/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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

// cuda::std::ranges::ssize

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

using RangeSSizeT = decltype(cuda::std::ranges::ssize);

static_assert(!cuda::std::is_invocable_v<RangeSSizeT, int[]>);
static_assert(cuda::std::is_invocable_v<RangeSSizeT, int[1]>);
static_assert(cuda::std::is_invocable_v<RangeSSizeT, int (&&)[1]>);
static_assert(cuda::std::is_invocable_v<RangeSSizeT, int (&)[1]>);

struct SizeMember
{
  __host__ __device__ constexpr size_t size()
  {
    return 42;
  }
};
static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::ssize), const SizeMember&>);

struct SizeFunction
{
  __host__ __device__ friend constexpr size_t size(SizeFunction)
  {
    return 42;
  }
};

struct SizeFunctionSigned
{
  __host__ __device__ friend constexpr cuda::std::ptrdiff_t size(SizeFunctionSigned)
  {
    return 42;
  }
};

struct SizedSentinelRange
{
  int data_[2] = {};
  __host__ __device__ constexpr int* begin()
  {
    return data_;
  }
  __host__ __device__ constexpr auto end()
  {
    return sized_sentinel<int*>(data_ + 2);
  }
};

struct ShortUnsignedReturnType
{
  __host__ __device__ constexpr unsigned short size()
  {
    return 42;
  }
};

// size_t changes depending on the platform.
using SignedSizeT = cuda::std::make_signed_t<size_t>;

__host__ __device__ constexpr bool test()
{
  int a[4] = {};

  assert(cuda::std::ranges::ssize(a) == 4);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::ssize(a)), SignedSizeT>);

  assert(cuda::std::ranges::ssize(SizeMember()) == 42);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::ssize(SizeMember())), SignedSizeT>);

  assert(cuda::std::ranges::ssize(SizeFunction()) == 42);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::ssize(SizeFunction())), SignedSizeT>);

  assert(cuda::std::ranges::ssize(SizeFunctionSigned()) == 42);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::ssize(SizeFunctionSigned())), cuda::std::ptrdiff_t>);

  SizedSentinelRange b{};
  assert(cuda::std::ranges::ssize(b) == 2);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::ssize(b)), cuda::std::ptrdiff_t>);

  // This gets converted to ptrdiff_t because it's wider.
  ShortUnsignedReturnType c{};
  assert(cuda::std::ranges::ssize(c) == 42);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::ssize(c)), ptrdiff_t>);

  return true;
}

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
static_assert(!cuda::std::is_invocable_v<RangeSSizeT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeSSizeT, Holder<Incomplete>*&>);
#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
