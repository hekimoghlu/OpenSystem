/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <uscl/cmath>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/limits>
#include <uscl/std/utility>

#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <cstdint>
#endif // !TEST_COMPILER(NVRTC)

template <class T, class U>
__host__ __device__ constexpr void test()
{
  constexpr T maxv = cuda::std::numeric_limits<T>::max();

  // ensure that we return the right type
  using Common = ::cuda::std::common_type_t<T, U>;
  static_assert(cuda::std::is_same<decltype(cuda::ceil_div(T(0), U(1))), Common>::value);
  assert(cuda::ceil_div(T(0), U(1)) == Common(0));
  assert(cuda::ceil_div(T(1), U(1)) == Common(1));
  assert(cuda::ceil_div(T(126), U(64)) == Common(2));

  // ensure that we are resilient against overflow
  assert(cuda::ceil_div(maxv, U(1)) == maxv);
  assert(cuda::ceil_div(maxv, maxv) == Common(1));
}

template <class T>
__host__ __device__ constexpr void test()
{
  // Builtin integer types:
  test<T, char>();
  test<T, signed char>();
  test<T, unsigned char>();

  test<T, short>();
  test<T, unsigned short>();

  test<T, int>();
  test<T, unsigned int>();

  test<T, long>();
  test<T, unsigned long>();

  test<T, long long>();
  test<T, unsigned long long>();

#if !TEST_COMPILER(NVRTC)
  // cstdint types:
  test<T, std::size_t>();
  test<T, std::ptrdiff_t>();
  test<T, std::intptr_t>();
  test<T, std::uintptr_t>();

  test<T, std::int8_t>();
  test<T, std::int16_t>();
  test<T, std::int32_t>();
  test<T, std::int64_t>();

  test<T, std::uint8_t>();
  test<T, std::uint16_t>();
  test<T, std::uint32_t>();
  test<T, std::uint64_t>();
#endif // !TEST_COMPILER(NVRTC)

#if _CCCL_HAS_INT128()
  test<T, __int128_t>();
  test<T, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr bool test()
{
  // Builtin integer types:
  test<char>();
  test<signed char>();
  test<unsigned char>();

  test<short>();
  test<unsigned short>();

  test<int>();
  test<unsigned int>();

  test<long>();
  test<unsigned long>();

  test<long long>();
  test<unsigned long long>();

#if !TEST_COMPILER(NVRTC)
  // cstdint types:
  test<std::size_t>();
  test<std::ptrdiff_t>();
  test<std::intptr_t>();
  test<std::uintptr_t>();

  test<std::int8_t>();
  test<std::int16_t>();
  test<std::int32_t>();
  test<std::int64_t>();

  test<std::uint8_t>();
  test<std::uint16_t>();
  test<std::uint32_t>();
  test<std::uint64_t>();
#endif // !TEST_COMPILER(NVRTC)

#if _CCCL_HAS_INT128()
  test<__int128_t>();
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test(), "");
  return 0;
}
