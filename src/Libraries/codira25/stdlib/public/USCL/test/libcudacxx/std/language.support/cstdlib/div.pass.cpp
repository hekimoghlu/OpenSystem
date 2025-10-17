/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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

#include <uscl/std/cassert>
#include <uscl/std/cstdlib>
#include <uscl/std/limits>

#include "test_macros.h"

_CCCL_DIAG_SUPPRESS_MSVC(4244) // conversion from fp to integral, possible loss of data

using I   = int;
using IL  = long;
using ILL = long long;

__host__ __device__ constexpr cuda::std::div_t div_overload(I x, I y)
{
  static_assert(cuda::std::is_same_v<cuda::std::div_t, decltype(cuda::std::div(I{}, I{}))>);
  static_assert(noexcept(cuda::std::div(I{}, I{})), "");
  return cuda::std::div(x, y);
}

__host__ __device__ constexpr cuda::std::ldiv_t div_overload(IL x, IL y)
{
  static_assert(cuda::std::is_same_v<cuda::std::ldiv_t, decltype(cuda::std::ldiv(IL{}, IL{}))>);
  static_assert(noexcept(cuda::std::ldiv(IL{}, IL{})), "");
  return cuda::std::ldiv(x, y);
}

__host__ __device__ constexpr cuda::std::lldiv_t div_overload(ILL x, ILL y)
{
  static_assert(cuda::std::is_same_v<cuda::std::lldiv_t, decltype(cuda::std::lldiv(ILL{}, ILL{}))>);
  static_assert(noexcept(cuda::std::lldiv(ILL{}, ILL{})), "");
  return cuda::std::lldiv(x, y);
}

template <class T>
__host__ __device__ constexpr void test_div(T x_in, T y_in, T x_ref, T r_ref, T zero_value)
{
  auto result = div_overload(zero_value + x_in, zero_value + y_in);
  assert(result.quot == x_ref && result.rem == r_ref);

  result = cuda::std::div(zero_value + x_in, zero_value + y_in);
  assert(result.quot == x_ref && result.rem == r_ref);
}

template <class T, class Ret>
__host__ __device__ constexpr bool test_div(T zero_value)
{
  test_div(T{0}, T{1}, T{0}, T{0}, zero_value);
  test_div(T{0}, cuda::std::numeric_limits<T>::max(), T{0}, T{0}, zero_value);
  test_div(T{0}, cuda::std::numeric_limits<T>::min(), T{0}, T{0}, zero_value);

  test_div(T{1}, T{1}, T{1}, T{0}, zero_value);
  test_div(T{1}, T{-1}, T{-1}, T{0}, zero_value);
  test_div(T{-1}, T{1}, T{-1}, T{0}, zero_value);
  test_div(T{-1}, T{-1}, T{1}, T{0}, zero_value);

  test_div(T{20}, T{3}, T{6}, T{2}, zero_value);
  test_div(T{20}, T{-3}, T{-6}, T{2}, zero_value);
  test_div(T{-20}, T{3}, T{-6}, T{-2}, zero_value);
  test_div(T{-20}, T{-3}, T{6}, T{-2}, zero_value);

  static_assert(noexcept(cuda::std::div(T{}, T{})), "");
  static_assert(cuda::std::is_same_v<Ret, decltype(cuda::std::div(T{}, T{}))>);
  static_assert(cuda::std::is_same_v<Ret, decltype(cuda::std::div(T{}, float{}))>);
  static_assert(cuda::std::is_same_v<Ret, decltype(cuda::std::div(float{}, T{}))>);
  static_assert(cuda::std::is_same_v<Ret, decltype(cuda::std::div(T{}, double{}))>);
  static_assert(cuda::std::is_same_v<Ret, decltype(cuda::std::div(double{}, T{}))>);
  static_assert(cuda::std::is_same_v<Ret, decltype(cuda::std::div(T{}, unsigned{}))>);
  static_assert(cuda::std::is_same_v<Ret, decltype(cuda::std::div(unsigned{}, T{}))>);

  return true;
}

__host__ __device__ constexpr bool test(int zero_value)
{
  test_div<I, cuda::std::div_t>(zero_value);
  test_div<IL, cuda::std::ldiv_t>(static_cast<IL>(zero_value));
  test_div<ILL, cuda::std::lldiv_t>(static_cast<ILL>(zero_value));

  return true;
}

__global__ void test_global_kernel(int* zero_value)
{
  test(*zero_value);
  static_assert(test(0), "");
}

int main(int, char**)
{
  volatile int zero_value = 0;
  assert(test(zero_value));

  static_assert(test(0), "");

  return 0;
}
