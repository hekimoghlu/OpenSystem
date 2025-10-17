/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
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
  assert(cuda::bitfield_insert(T{0}, all_ones, 0, 1) == 1);
  assert(cuda::bitfield_insert(T{0}, all_ones, 1, 1) == 0b10);
  assert(cuda::bitfield_insert(T{0b10}, all_ones, 0, 1) == 0b11);
  assert(cuda::bitfield_insert(all_ones, all_ones, 0, 0) == all_ones);
  assert(cuda::bitfield_insert(all_ones, all_ones, 0, 1) == all_ones);
  assert(cuda::bitfield_insert(all_ones, all_ones, 2, 1) == all_ones);
  assert(cuda::bitfield_insert(all_ones, T{0b1000}, 1, 2) == (all_ones & static_cast<T>(~T{0b110})));

  assert(cuda::bitfield_insert(T{0}, all_ones, 0, 2) == 0b11);
  assert(cuda::bitfield_insert(T{0}, all_ones, 3, 2) == 0b11000);
  assert(cuda::bitfield_insert(T{0b10100000}, all_ones, 3, 2) == 0b10111000);
  assert(cuda::bitfield_insert(T{0b10100000}, T{0b11}, 3, 2) == 0b10111000);
  assert(cuda::bitfield_insert(T{0}, all_ones, nl::digits - 1, 1) == (T{1} << (nl::digits - 1u)));
  assert(cuda::bitfield_insert(T{0b10100000}, all_ones, 0, nl::digits) == all_ones);
  assert(cuda::bitfield_insert(T{0b10100000}, all_ones, nl::digits, 0) == T{0b10100000});

  assert(cuda::bitfield_extract(T{0}, 3, 4) == 0);
  assert(cuda::bitfield_extract(T{0b1011}, 0, 1) == 1);
  assert(cuda::bitfield_extract(T{0b1011}, 1, 1) == 1);
  assert(cuda::bitfield_extract(T{0b1011}, 2, 2) == 0b10);
  assert(cuda::bitfield_extract(all_ones, 0, 0) == 0);
  assert(cuda::bitfield_extract(all_ones, 0, 4) == 0b1111);
  assert(cuda::bitfield_extract(all_ones, 2, 4) == 0b1111);

  assert(cuda::bitfield_extract(T{0b1010010}, 0, 2) == 0b10);
  assert(cuda::bitfield_extract(T{0b10101100}, 3, 2) == 1);
  assert(cuda::bitfield_extract(T{0b10100000}, 3, 3) == 0b100);

  assert(cuda::bitfield_extract(T{all_ones}, nl::digits - 1, 1) == 1);
  assert(cuda::bitfield_extract(T{0b10100000}, 0, nl::digits) == T{0b10100000});
  assert(cuda::bitfield_extract(T{0b10100000}, nl::digits, 0) == 0);
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
