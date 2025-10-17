/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <uscl/bit>
#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr bool test()
{
  using nl             = cuda::std::numeric_limits<T>;
  constexpr T all_ones = static_cast<T>(~T{0});
  unused(all_ones);
  assert(cuda::bitmask<T>(0, 0) == 0);
  assert(cuda::bitmask<T>(0, 1) == 1);
  assert(cuda::bitmask<T>(1, 0) == 0);
  assert(cuda::bitmask<T>(1, 1) == 0b10);
  assert(cuda::bitmask<T>(0, 2) == 0b11);
  assert(cuda::bitmask<T>(2, 2) == 0b1100);

  assert(cuda::bitmask<T>(0, 2) == 0b11);
  assert(cuda::bitmask<T>(3, 2) == 0b11000);
  assert(cuda::bitmask<T>(nl::digits - 1, 1) == (T{1} << (nl::digits - 1u)));
  assert(cuda::bitmask<T>(0, nl::digits) == all_ones);
  assert(cuda::bitmask<T>(nl::digits, 0) == 0);
  return true;
}

__host__ __device__ constexpr bool test()
{
  test<unsigned char>();
  test<unsigned short>();
  test<unsigned>();
  test<unsigned long>();
  test<unsigned long long>();

  test<uint8_t>();
  test<uint16_t>();
  test<uint32_t>();
  test<uint64_t>();
  test<size_t>();
  test<uintmax_t>();
  test<uintptr_t>();

#if _CCCL_HAS_INT128()
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
