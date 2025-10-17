/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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

using I   = int;
using IL  = long;
using ILL = long long;

__host__ __device__ constexpr I abs_overload(I value)
{
  static_assert(cuda::std::is_same_v<I, decltype(cuda::std::abs(I{}))>);
  static_assert(noexcept(cuda::std::abs(I{})), "");
  return cuda::std::abs(value);
}

__host__ __device__ constexpr IL abs_overload(IL value)
{
  static_assert(cuda::std::is_same_v<IL, decltype(cuda::std::labs(IL{}))>);
  static_assert(noexcept(cuda::std::labs(IL{})), "");
  return cuda::std::labs(value);
}

__host__ __device__ constexpr ILL abs_overload(ILL value)
{
  static_assert(cuda::std::is_same_v<ILL, decltype(cuda::std::llabs(ILL{}))>);
  static_assert(noexcept(cuda::std::llabs(ILL{})), "");
  return cuda::std::llabs(value);
}

template <class T>
__host__ __device__ constexpr bool test_abs(T in, T ref, T zero_value)
{
  assert(abs_overload(zero_value + in) == ref);
  assert(cuda::std::abs(zero_value + in) == ref);
  return true;
}

template <class T>
__host__ __device__ constexpr bool test_abs(T zero_value)
{
  test_abs(T{0}, T{0}, zero_value);
  test_abs(T{1}, T{1}, zero_value);
  test_abs(T{-1}, T{1}, zero_value);
  test_abs(T{257}, T{257}, zero_value);
  test_abs(T{-257}, T{257}, zero_value);
  test_abs(cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::max(), zero_value);
  test_abs(cuda::std::numeric_limits<T>::min() + T{1}, cuda::std::numeric_limits<T>::max(), zero_value);

  static_assert(noexcept(cuda::std::abs(T{})), "");
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::abs(T{}))>);

  return true;
}

__host__ __device__ constexpr bool test(int zero_value)
{
  test_abs(zero_value);
  test_abs(static_cast<IL>(zero_value));
  test_abs(static_cast<ILL>(zero_value));
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
