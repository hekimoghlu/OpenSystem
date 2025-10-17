/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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

// constexpr span() noexcept;

#include <uscl/std/cassert>
#include <uscl/std/span>
#include <uscl/std/type_traits>

#include "test_macros.h"

__host__ __device__ void checkCV()
{
  //  Types the same (dynamic sized)
  {
    cuda::std::span<int> s1;
    cuda::std::span<const int> s2;
    cuda::std::span<volatile int> s3;
    cuda::std::span<const volatile int> s4;
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
  }

  //  Types the same (static sized)
  {
    cuda::std::span<int, 0> s1;
    cuda::std::span<const int, 0> s2;
    cuda::std::span<volatile int, 0> s3;
    cuda::std::span<const volatile int, 0> s4;
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
  }
}

template <typename T>
__host__ __device__ constexpr bool testConstexprSpan()
{
  cuda::std::span<const T> s1;
  cuda::std::span<const T, 0> s2;
  return s1.data() == nullptr && s1.size() == 0 && s2.data() == nullptr && s2.size() == 0;
}

template <typename T>
__host__ __device__ void testRuntimeSpan()
{
  static_assert(noexcept(T{}));
  cuda::std::span<const T> s1;
  cuda::std::span<const T, 0> s2;
  assert(s1.data() == nullptr && s1.size() == 0);
  assert(s2.data() == nullptr && s2.size() == 0);
}

struct A
{};

int main(int, char**)
{
  static_assert(testConstexprSpan<int>());
  static_assert(testConstexprSpan<long>());
  static_assert(testConstexprSpan<double>());
  static_assert(testConstexprSpan<A>());

  testRuntimeSpan<int>();
  testRuntimeSpan<long>();
  testRuntimeSpan<double>();
  testRuntimeSpan<A>();

  checkCV();

  static_assert(cuda::std::is_default_constructible<cuda::std::span<int, cuda::std::dynamic_extent>>::value, "");
  static_assert(cuda::std::is_default_constructible<cuda::std::span<int, 0>>::value, "");
  static_assert(!cuda::std::is_default_constructible<cuda::std::span<int, 2>>::value, "");

  return 0;
}
